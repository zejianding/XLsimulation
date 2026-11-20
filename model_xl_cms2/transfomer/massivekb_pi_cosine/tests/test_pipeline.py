from __future__ import annotations
import ast
import csv
import gzip
import importlib.util
import io
import json
import lzma
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parents[1]
REPO=HERE.parents[2]
sys.path.insert(0,str(HERE))
from mgf_io import iter_blocks, parse_block_key, make_group_filename
from sample_massivekb_test import select_groups
from transform_mgf_intensity import transform_file
from pairwise_cosine import evaluate, load_group
from summary_all import summarize
from run_pipeline import Stages, validate_sample


def block(seq="PEPTIDE",sid="s1",charge=2,peaks="100 100",title="duplicate-title"):
    return f"BEGIN IONS\nTITLE={title}\nSPECTRUM_ID={sid}\nSEQ={seq}\nCHARGE={charge}+\nPEPMASS=400\n{peaks}\nEND IONS\n"


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


class RegressionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def run_cli(self,script,*args,ok=True):
        r=subprocess.run([sys.executable,str(script),*map(str,args)],capture_output=True,text=True,encoding="utf-8",errors="replace")
        if ok: self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        else: self.assertNotEqual(r.returncode,0)
        return r

    def test_preferred_sampling_and_fallback(self):
        counts={("A",2):3,("B",2):7,("C",2):2,("D",2):1}
        picked,npref,nfall=select_groups(counts,2,99)
        self.assertEqual(set(picked),{("A",2),("B",2)})
        self.assertEqual(select_groups(counts,3,99)[0],[("A",2),("B",2),("C",2)])
        self.assertEqual((npref,nfall),(2,1))
        with self.assertRaises(ValueError): select_groups(counts,4,99)

    def test_unique_prosit_sampling_keeps_charge_and_one_mod_form(self):
        from prosit_sampling import candidates, backbone
        counts={("PEPTIDE",2):3,("+42.011PEPTIDE",2):5,
                ("PEPTIDE",3):4,("PEPTIDEM+15.995",2):2,
                ("PEPTIDEM",2):8,("A"*31,2):20,("PEPXIDE",2):5}
        high,low,rejected=candidates(counts,42,2,3)
        self.assertEqual(len(high),3)
        self.assertEqual(low,[])
        self.assertEqual({(backbone(s),c) for s,c in high},
                         {('PEPTIDE',2),('PEPTIDE',3),('PEPTIDEM',2)})
        self.assertIn(('PEPTIDEM',2),high)
        self.assertEqual(candidates(counts,42,2,3),(high,low,rejected))

    def test_unique_prosit_cli_and_collision_validation(self):
        source=self.root/'input.mgf'
        seqs=['PEPTIDE','+42.011PEPTIDE','PEPTIDEM','A'*31]
        source.write_text(''.join(block(s,str(i)).replace(f'SPECTRUM_ID={i}\n','')
                                  for s in seqs for i in range(3)))
        out=self.root/'unique'
        self.run_cli(HERE/'sample_massivekb_test.py','--mgf-xz',source,'--out-dir',out,
                     '--n-groups',2,'--min-spectra',3,'--unique-prosit-input')
        counts,ids=validate_sample(out,2,3)
        self.assertEqual(sum(counts.values()),6)
        meta=json.loads((out/'massTest_meta.json').read_text())
        self.assertEqual(meta['prosit_sampling']['n_unique_prosit_inputs'],2)
        self.assertFalse(meta['prosit_sampling']['prediction_verified'])

    def test_verified_unique_sampling_replaces_failed_candidate(self):
        import prosit_sampling as ps
        counts={(s,2):3 for s in ['PEPTIDE','AAAAAAA','GGGGGGG','PEPTIDEM']}
        high,_,_=ps.candidates(counts,12,3,3)
        failed=high[0]
        out=self.root/'verified';out.mkdir()
        def predict(cmd,check):
            folder=Path(cmd[cmd.index('--out')+1]).parent
            with np.load(folder/'inputs.npz') as z:
                keys=list(zip(z['seqs'].tolist(),z['charges'].tolist()))
            (folder/'predicted.mgf').write_text(''.join(block(s,s,c) for s,c in keys if (s,c)!=failed))
            with (folder/'skipped.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['sequence','charge','reason','detail'])
                w.writerows((s,c,'empty_prediction','') for s,c in keys if (s,c)==failed)
        with mock.patch.object(ps.subprocess,'run',side_effect=predict):
            picked,_,_,report=ps.select_unique(counts,2,12,3,3,out,verify=True,batch_size=2)
        self.assertEqual(set(picked),set(high[1:3]))
        self.assertEqual(report['prediction_attempted_groups'],3)
        self.assertEqual(report['prediction_rejected_groups'],1)
        self.assertEqual(report['prediction_successful_groups'],2)
        with (out/'prosit_predicted.mgf.partial').open('rb') as f:
            self.assertEqual({parse_block_key(b) for b in iter_blocks(f)},set(picked))

    def test_sampling_full_groups_ids_and_no_overwrite(self):
        source=self.root/"input.xz"
        text="".join(block(seq,str(i)).replace(f"SPECTRUM_ID={i}\n","") for seq,n in [("PEPTIDE",3),("AAAAAAA",2),("GGGGGGG",1)] for i in range(n))
        source.write_bytes(lzma.compress(text.encode()))
        out=self.root/"sample"
        self.run_cli(HERE/"sample_massivekb_test.py","--mgf-xz",source,"--out-dir",out,"--n-groups",2,"--threads",1)
        counts,ids=validate_sample(out,2,3)
        self.assertEqual(sum(counts.values()),5);self.assertEqual(len(ids),5)
        meta=json.loads((out/"massTest_meta.json").read_text())
        self.assertEqual(meta["n_fallback_selected"],1)
        original=(out/"massTest.mgf").read_bytes()
        self.run_cli(HERE/"sample_massivekb_test.py","--mgf-xz",source,"--out-dir",out,"--n-groups",2,ok=False)
        self.assertEqual(original,(out/"massTest.mgf").read_bytes())

    def test_intensity_fraction_exponent_and_none(self):
        src=self.root/"in.mgf";src.write_text(block(peaks="100 0.25\n2e2 1.5e4\n300 2.9"))
        dst=self.root/"out.mgf";transform_file(src,dst)
        peaks=[float(x.split()[1]) for x in dst.read_text().splitlines() if x[:1].isdigit()]
        np.testing.assert_allclose(peaks,np.sqrt(np.array([.25,15000,2.9])/50000),rtol=1e-10)
        transform_file(src,dst,"none");self.assertIn("2e2 1.5e4",dst.read_text())
        with self.assertRaises(ValueError): transform_file(src,src)

    def test_parser_rejects_truncation_and_header_substrings(self):
        with self.assertRaises(ValueError):list(iter_blocks(io.BytesIO(b"BEGIN IONS\nSEQ=PEP\n")))
        self.assertIsNone(parse_block_key("BEGIN IONS\nTITLE=SEQ=ABC\nCHARGE=2+"))
        text=block().replace("\n","\r\n").encode()
        self.assertEqual(len(list(iter_blocks(io.BytesIO(text),read_size=1))),1)

    def create_groups(self,folder,seq,ids,emb,invalid=None,charge=2):
        folder.mkdir(exist_ok=True)
        name=make_group_filename(seq,charge)
        (folder/name).write_text("".join(block(seq,sid,charge) for sid in ids))
        np.savez(folder/(Path(name).stem+".npz"),embeddings=np.array(emb,float),spectrum_ids=np.array(ids),invalid_reasons=np.array(invalid or [""]*len(ids)))
        path=folder/"group_map.csv";exists=path.exists()
        with path.open("a",newline="") as fh:
            w=csv.writer(fh)
            if not exists:w.writerow(["group_file","sequence","charge","n_original","n_prediction"])
            w.writerow([name,seq,charge,len(ids),0])
        return name

    def fixture(self):
        paths={k:self.root/k for k in ["r","f","ar","af","q"]}
        name=self.create_groups(paths["r"],"PEPTIDE",["A","B"],[[1,0],[0,1]])
        self.create_groups(paths["f"],"PEPTIDE",["B","A"],[[0,1],[1,0]])
        self.create_groups(paths["ar"],"PEPTIDE",["avg"],[[1,1]])
        self.create_groups(paths["af"],"PEPTIDE",["avg"],[[1,1]])
        self.create_groups(paths["q"],"PEPTIDE",["q"],[[1,0]])
        # A second group has NO Prosit prediction. Task 2 must still include it.
        for k in ("r","f"):
            self.create_groups(paths[k],"AAAAAAA",["C","D","E"],[[1,0]]*3)
        for k in ("ar","af"):
            self.create_groups(paths[k],"AAAAAAA",["avg2"],[[1,0]])
        a=types.SimpleNamespace(ref_grouped_dir=paths["r"],ref_emb_dir=paths["r"],src_grouped_dir=paths["f"],src_emb_dir=paths["f"],avg_ref_grouped_dir=paths["ar"],avg_ref_emb_dir=paths["ar"],avg_src_grouped_dir=paths["af"],avg_src_emb_dir=paths["af"],prosit_grouped_dir=paths["q"],prosit_emb_dir=paths["q"],domain="none",out_dir=self.root/"metrics")
        return paths,name,a

    def test_all_metrics_and_identity_alignment(self):
        paths,name,a=self.fixture();evaluate(a)
        m=pd.read_csv(Path(a.out_dir)/"group_metrics.csv")
        selected=m[m.group_file==name].set_index(["source","mode"])
        self.assertAlmostEqual(selected.loc[("filtered","direct"),"mean"],1)
        self.assertAlmostEqual(selected.loc[("filtered","pairwise"),"mean"],.5)
        self.assertAlmostEqual(selected.loc[("filtered","averaged"),"mean"],1)
        self.assertAlmostEqual(selected.loc[("filtered","averaged_each"),"mean"],2**-.5)
        self.assertEqual(selected.loc[("baseline","real_real"),"mean"],0)
        self.assertEqual(selected.loc[("prosit","direct"),"mean"],selected.loc[("prosit","pairwise"),"mean"])
        summary=summarize([Path(a.out_dir)/"group_metrics.csv"])
        rr=summary[(summary.source=="baseline")&(summary["mode"]=="real_real")&(summary.layer=="all")]
        self.assertEqual(rr[rr.weighting=="group"].iloc[0].n_groups,2)
        self.assertAlmostEqual(rr[rr.weighting=="group"].iloc[0]["mean"],.5)
        self.assertAlmostEqual(rr[rr.weighting=="pooled"].iloc[0]["mean"],.75)
        cross=summary[(summary.source=="filtered")&(summary["mode"]=="pairwise")&(summary.layer=="all")&(summary.weighting=="pooled")].iloc[0]
        self.assertEqual(cross.n_units,13);self.assertAlmostEqual(cross["mean"],11/13)
        paired=summary[(summary.layer=="paired")&(summary["mode"]=="direct")&(summary.weighting=="group")]
        self.assertEqual(paired.n_groups.tolist(),[1,1])
        with gzip.open(Path(a.out_dir)/"mode3_pairs.csv.gz","rt") as fh:self.assertEqual(len(list(csv.DictReader(fh))),13)
        with gzip.open(Path(a.out_dir)/"task2_pairs.csv.gz","rt") as fh:self.assertEqual(len(list(csv.DictReader(fh))),4)

    def test_invalid_filtered_does_not_remove_real_baseline(self):
        paths,name,a=self.fixture()
        np.savez(paths["f"]/(Path(name).stem+".npz"),embeddings=np.array([[np.nan,np.nan],[1,0]]),spectrum_ids=np.array(["B","A"]),invalid_reasons=np.array(["no_usable_peaks",""]))
        evaluate(a);s=summarize([Path(a.out_dir)/"group_metrics.csv"])
        self.assertTrue((s[(s.layer=="paired")&(s["mode"]=="direct")].n_groups==0).all())
        self.assertTrue((s[(s.source=="baseline")&(s["mode"]=="real_real")&(s.layer=="all")].n_groups==2).all())

    def test_missing_and_duplicate_embedding_ids_fail(self):
        paths,name,a=self.fixture();npz=paths["f"]/(Path(name).stem+".npz")
        np.savez(npz,embeddings=np.eye(2),spectrum_ids=np.array(["A","A"]))
        with self.assertRaises(ValueError):evaluate(a)
        npz.unlink()
        with self.assertRaises(FileNotFoundError):load_group(paths["f"]/name,npz,2)

    def test_prosit_annotation_alignment_and_cache_roundtrip(self):
        path=HERE.parent/"ds/step4_prosit_score.py"
        tree=ast.parse(path.read_text(encoding="utf8"))
        wanted={"_sorted_peaks","parse_annotation","PrositPredictor"}
        body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in wanted]
        ns=dict(np=np,pd=pd,Path=Path,re=re,gzip=gzip,sys=sys)
        exec(compile(ast.Module(body=body,type_ignores=[]),str(path),"exec"),ns)
        cache=self.root/"cache.tsv.gz"
        cache.write_text("legacy poisoned cache is ignored")
        obj=ns["PrositPredictor"](cache)
        obj._absorb(pd.DataFrame(dict(peptide_sequences=["PEPTIDE"]*2,precursor_charges=[2]*2,mz=[200.,100.],intensities=[.8,.2],annotation=["b2","b1"])))
        arr,annos=obj.get("PEPTIDE",2)
        self.assertEqual(annos,["b1","b2"]);np.testing.assert_allclose(arr[:,1],[.2,.8])
        obj.save_cache();obj2=ns["PrositPredictor"](cache)
        arr2,annos2=obj2.get("PEPTIDE",2)
        np.testing.assert_allclose(arr,arr2);self.assertEqual(annos,annos2)

    def test_stage_resume_corruption_failure_and_parameters(self):
        root=self.root/"pipeline";st=Stages(root)
        calls=[]
        def action(d):calls.append(1);(d/"value").write_text("correct")
        def check(d):self.assertEqual((d/"value").read_text(),"correct")
        first=st.run("unit",dict(tol=.05),[],action,check)
        self.assertEqual(st.run("unit",dict(tol=.05),[],action,check),first)
        (first/"value").write_text("corrupt")
        second=st.run("unit",dict(tol=.05),[],action,check)
        self.assertNotEqual(first,second);self.assertTrue(first.exists())
        third=st.run("unit",dict(tol=.1),[],action,check);self.assertNotEqual(second,third)
        def fail(d):(d/"partial").write_text("unfinished");raise RuntimeError("interrupted")
        with self.assertRaises(RuntimeError):st.run("broken",{},[],fail,check)
        self.assertFalse((root/"state/broken.json").exists())
        (third/"stage_manifest.json").unlink()
        fourth=st.run("unit",dict(tol=.1),[],action,check)
        self.assertNotEqual(third,fourth)
        self.assertEqual(len(calls),4)

    def test_embedding_worker_failure_is_not_success(self):
        mod=load_module("embedding_batch_test",REPO/"piprime_embedding/piprime_embedding/mgf_to_embedding_batch.py")
        inputs=self.root/"mgfs";inputs.mkdir();(inputs/"one.mgf").write_text(block())
        with mock.patch.object(mod,"process_single_file",side_effect=RuntimeError("bad model")):
            with self.assertRaises(RuntimeError):mod.process_folder(str(inputs),str(self.root/"npzs"),None,None,None,None)

    def test_external_prediction_partition_includes_charge(self):
        source=self.root/"charges.xz"
        text="".join(block(charge=c,sid=str(i)).replace(f"SPECTRUM_ID={i}\n","") for c in (2,3) for i in range(3))
        source.write_bytes(lzma.compress(text.encode()))
        prediction=self.root/"pred.mgf";prediction.write_text(block(charge=2))
        out=self.root/"run_charge"
        self.run_cli(HERE/"run_pipeline.py","--out-dir",out,"--train-mgf-xz",source,"--n-groups",2,"--threads",1,"--workers",1,"--prosit-mgf",prediction,"--stop-after","prepare")
        report=json.loads((out/"run_manifest.json").read_text(encoding="utf8"))
        skip=pd.read_csv(Path(report["paths"]["prosit"])/"prosit_skipped.csv")
        self.assertEqual(skip.charge.tolist(),[3])
        self.assertEqual(skip.sequence.tolist(),["PEPTIDE"])

    def test_embedding_identity_metadata_survives_mgf_reader(self):
        mod=load_module("embedding_reader_test",REPO/"piprime_embedding/piprime_embedding/mgf_to_embedding_batch.py")
        mgf=self.root/"identity.mgf";mgf.write_text(block(sid="stable-1")+block(sid="stable-2"))
        fake=types.SimpleNamespace(extract_embedding=lambda *args:np.array([1.,2.]))
        result=mod.extract_embeddings_from_one_mgf(str(mgf),fake,None,{"dim_model":2},None)
        self.assertEqual(result[5].tolist(),["stable-1","stable-2"])
        self.assertEqual(result[6].tolist(),["",""])

    def test_encoder_only_loads_exact_weights_and_mean_pooling(self):
        import torch
        embed_dir=REPO/"piprime_embedding/piprime_embedding"
        sys.path.insert(0,str(embed_dir))
        from PrimeNovo.components import SpectrumEncoder
        import encoder_only
        hp=dict(dim_model=16,n_head=4,dim_feedforward=32,n_layers=1,dropout=0.,dim_intensity=None,custom_encoder=None)
        original=SpectrumEncoder(**{k:v for k,v in hp.items() if k!="custom_encoder"}).eval()
        ckpt=self.root/"encoder.ckpt"
        torch.save(dict(hyper_parameters=hp,state_dict={"encoder."+k:v for k,v in original.state_dict().items()}),ckpt)
        with mock.patch.object(torch.cuda,"is_available",return_value=False):
            loaded,config,device=encoder_only.load_model(ckpt)
        peaks=torch.tensor([[100.,.2],[200.,.8]])
        with torch.no_grad():expected=original(peaks.unsqueeze(0))[0].mean(dim=1).numpy()[0]
        with mock.patch.object(encoder_only,"process_peaks",return_value=peaks):
            got=encoder_only.extract_embedding(None,None,400.,2,loaded,config,device)
        np.testing.assert_allclose(got,expected,rtol=0,atol=0)
        self.assertEqual(config["dim_model"],16)

    def test_actual_prepare_pipeline_and_resume(self):
        source=self.root/"input.xz"
        text="".join(block(seq,str(i)).replace(f"SPECTRUM_ID={i}\n","") for seq,n in [("PEPTIDE",3),("AAAAAAA",3)] for i in range(n))
        source.write_bytes(lzma.compress(text.encode()))
        out=self.root/"run"
        args=["--out-dir",out,"--train-mgf-xz",source,"--n-groups",2,"--threads",1,"--workers",2,"--skip-prosit","--stop-after","prepare"]
        self.run_cli(HERE/"run_pipeline.py",*args)
        report=json.loads((out/"run_manifest.json").read_text(encoding="utf8"))
        self.assertEqual(report["status"],"prepare_complete")
        resumed=self.run_cli(HERE/"run_pipeline.py",*args)
        self.assertIn("REUSE sample",resumed.stdout);self.assertIn("REUSE filtered",resumed.stdout)


if __name__=="__main__":unittest.main()
