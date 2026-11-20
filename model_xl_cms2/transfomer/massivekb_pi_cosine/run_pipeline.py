#!/usr/bin/env python3
"""Transactional massiveKB analysis. See ????.md for the output contract."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
import numpy as np
from mgf_io import iter_blocks, parse_block_key, headers_of, make_group_filename
from pairwise_cosine import load_group, read_groups

HERE = Path(__file__).resolve().parent
TF = HERE.parent
REPO = TF.parent.parent
EMBED = REPO/"piprime_embedding/piprime_embedding/mgf_to_embedding_batch.py"
FILTER = TF.parent/"dataset/TokenFilter_parallel.py"
AVERAGE = TF/"eval_10k_gt/grouped_to_average_mgf.py"
PROSIT = TF/"eval_10k_gt/prosit_predict_mgf.py"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda:fh.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path); tmp = path.with_name(path.name+".partial")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def inventory(root):
    return {str(p.relative_to(root)):dict(size=p.stat().st_size, sha256=sha(p))
            for p in sorted(Path(root).rglob("*")) if p.is_file() and p.name != "stage_manifest.json"}


def dependency(path):
    p = Path(path).resolve()
    if p.is_dir():
        return dict(path=str(p), manifest=sha(p/"stage_manifest.json"))
    st = p.stat()
    # Source xz is huge: use stable file metadata, and recheck after both passes.
    return dict(path=str(p), size=st.st_size, mtime_ns=st.st_mtime_ns,
                sha256=sha(p) if p.suffix != ".xz" else None)


class Stages:
    def __init__(self, root, force=False):
        self.root = Path(root).resolve(); self.force = force
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root/"state").mkdir(exist_ok=True); (self.root/"logs").mkdir(exist_ok=True)
        sources = list(HERE.glob("*.py")) + [FILTER, AVERAGE, PROSIT, TF/"ds/step4_prosit_score.py"]
        sources += list((TF.parent/"dataset").glob("*.py"))
        sources += list(EMBED.parent.rglob("*.py")) + [EMBED.parent/"PrimeNovo/config.yaml"]
        self.code = {str(p.relative_to(REPO)):sha(p) for p in sorted(set(sources))}
        self.versions = {}
        for name in ("numpy", "pandas", "torch", "pyteomics", "spectrum_utils", "koinapy", "pytorch-lightning"):
            try: self.versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: self.versions[name] = "missing"
        self.paths = {}

    def command(self, script, *args):
        cmd = [sys.executable, "-u", str(script), *map(str,args)]
        print("RUN",subprocess.list2cmdline(cmd),flush=True)
        with (self.root/"logs"/"pipeline.log").open("a",encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace",
                                    env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            for line in proc.stdout:
                log.write(line); log.flush()
                print(line,end="",flush=True)
            if proc.wait():
                raise RuntimeError(f"Command failed: {script}; see pipeline.log")

    def run(self, name, config, inputs, action, validate):
        spec = dict(config=config, inputs=[dependency(x) for x in inputs], code=self.code, versions=self.versions)
        key = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
        state_path = self.root/"state"/(name+".json")
        if state_path.exists() and not self.force:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8")); target=Path(state["path"])
                if state["key"] == key and target.exists():
                    manifest = json.loads((target/"stage_manifest.json").read_text(encoding="utf-8"))
                    if manifest["key"] == key and manifest["files"] == inventory(target):
                        print("REUSE",name,flush=True); self.paths[name]=str(target); return target
                    print("REBUILD",name,"(output inventory changed)",flush=True)
            except (OSError, ValueError, KeyError, TypeError):
                print("REBUILD",name,"(missing or invalid completion state)",flush=True)
        work = Path(tempfile.mkdtemp(prefix=".partial-"+name+"-",dir=self.root))
        action(work)
        validate(work)
        if [dependency(x) for x in inputs] != spec["inputs"]:
            raise RuntimeError("Stage input changed during execution")
        json_write(work/"stage_manifest.json",dict(key=key, spec=spec, files=inventory(work)))
        target = self.root/(name+"-"+key[:10]+"-"+uuid.uuid4().hex[:8]); work.rename(target)
        json_write(state_path,dict(key=key,path=str(target)))
        self.paths[name]=str(target)
        return target


def scan(path, require_ids=False):
    counts=Counter(); identities={}
    with Path(path).open("rb") as fh:
        for block in iter_blocks(fh):
            key = parse_block_key(block)
            if key is None: raise ValueError(f"Missing group key: {path}")
            counts[key]+=1
            if require_ids:
                sid=headers_of(block).get("SPECTRUM_ID")
                if not sid or sid in identities: raise ValueError(f"Missing/duplicate spectrum ID: {path}")
                identities[sid]=key
    return counts,identities


def validate_sample(folder, n_groups=None, prefer=None):
    folder=Path(folder)
    meta=json.loads((folder/"massTest_meta.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != 2:
        raise ValueError("Legacy sample lacks identity/preferred sampling; resample into a new directory")
    with (folder/"massTest_groups.tsv").open(encoding="utf-8") as fh:
        rows=list(csv.DictReader(fh,delimiter="\t"))
    expected={(r["sequence"],int(r["charge"])):int(r["n_spectra"]) for r in rows}
    counts,ids=scan(folder/"massTest.mgf",True)
    if len(expected)!=len(rows) or counts!=expected or any(int(r["emitted"])!=int(r["n_spectra"]) for r in rows):
        raise ValueError("Sample manifest/spectrum counts differ")
    if n_groups is not None and len(expected)!=n_groups: raise ValueError("Incorrect sample group count")
    if prefer is not None and meta["prefer_spectra"]!=prefer: raise ValueError("Sample preference differs")
    if meta["n_spectra_emitted"]!=len(ids) or meta["n_groups_selected"]!=len(expected):
        raise ValueError("Sample metadata differs")
    fallback=sum(n<meta["prefer_spectra"] for n in counts.values())
    if fallback!=meta["n_fallback_selected"] or fallback!=max(0,len(expected)-meta["preferred_pool_size"]):
        raise ValueError("Sample did not satisfy preferred-stratum policy")
    if meta.get("prosit_sampling"):
        from prosit_sampling import backbone
        report = meta["prosit_sampling"]
        inputs = {(backbone(s),c) for s,c in expected}
        if len(inputs)!=len(expected) or report["n_unique_prosit_inputs"]!=len(expected):
            raise ValueError("Sample has duplicate unmodified sequence+charge inputs")
        if any(not 2<=len(s)<=30 or set(s)-set('ACDEFGHIKLMNPQRSTVWY') or c<1 for s,c in inputs):
            raise ValueError("Sample contains ineligible Prosit input")
        if report["prediction_verified"]:
            predictions,_=scan(folder/"prosit_predicted.mgf")
            if predictions!={key:1 for key in expected} or report["prediction_successful_groups"]!=len(expected):
                raise ValueError("Verified predictions do not cover all sample groups")
    return counts,ids


def validate_groups(folder, expected, prediction=False):
    rows=read_groups(folder)
    got={(r["sequence"],int(r["charge"])):int(r["n_prediction"] if prediction else r["n_original"]) for r in rows}
    if got!=dict(expected) or len(got)!=len(rows): raise ValueError("Grouped manifest mismatch")
    names={make_group_filename(s,c) for s,c in expected}
    if {p.name for p in Path(folder).glob("*.mgf")}!=names: raise ValueError("Grouped file set mismatch")
    for row in rows:
        key=(row["sequence"],int(row["charge"]))
        counts,_=scan(Path(folder)/row["group_file"])
        if counts!={key:expected[key]}: raise ValueError("Grouped spectrum count mismatch")


def validate_embeddings(folder, mgf_dir):
    expected={p.stem for p in Path(mgf_dir).glob("*.mgf")}
    if {p.stem for p in Path(folder).glob("*.npz")}!=expected: raise ValueError("Embedding file set mismatch")
    for stem in expected:
        counts,_=scan(Path(mgf_dir)/(stem+".mgf"))
        load_group(Path(mgf_dir)/(stem+".mgf"),Path(folder)/(stem+".npz"),sum(counts.values()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",required=True)
    p.add_argument("--train-mgf-xz")
    p.add_argument("--sample-dir",help="Explicitly reuse a schema-v2 sample; validates counts and IDs")
    p.add_argument("--model")
    p.add_argument("--n-groups",type=int,default=30000)
    p.add_argument("--min-spectra",type=int,default=2)
    p.add_argument("--prefer-spectra",type=int,default=3)
    p.add_argument("--seed",type=int,default=20261010)
    p.add_argument("--threads",type=int,default=0)
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--gpu",default="0")
    p.add_argument("--tol-filter",type=float,default=.05)
    p.add_argument("--tol-average",type=float,default=.05)
    p.add_argument("--domains",default="none,sqrt_clip",help="Both sources use the same transform within each domain")
    p.add_argument("--limit-groups",type=int,default=0,help="Sample this many COMPLETE groups for a smoke run; never truncate spectra")
    p.add_argument("--max-spectra",type=int,help="Explicit biased-prefix debug scan, marked in sample metadata")
    p.add_argument("--prosit-cache")
    p.add_argument("--prosit-mgf",help="Import predictions rather than call Koina; absent groups recorded as external_missing")
    p.add_argument("--skip-prosit",action="store_true",help="Run filtered + task2 only, explicitly mark prediction_missing")
    p.add_argument("--stop-after",choices=["sample","prepare","full"],default="full")
    p.add_argument("--force",action="store_true",help="Rebuild into new directories; preserve old artifacts")
    a=p.parse_args()
    if a.limit_groups<0 or a.workers<1 or a.tol_filter<=0 or a.tol_average<=0:
        p.error("Invalid limits/workers/tolerance")
    domains=a.domains.split(",")
    if not domains or len(set(domains))!=len(domains) or any(x not in ("none","sqrt_clip","sqrt") for x in domains):
        p.error("Invalid --domains")
    if a.skip_prosit and a.prosit_mgf: p.error("--skip-prosit conflicts with --prosit-mgf")
    if a.stop_after=="full" and (not a.model or not Path(a.model).is_file()): p.error("--model must name an existing checkpoint")
    if a.stop_after != "sample" and not a.skip_prosit and not a.prosit_mgf and importlib.util.find_spec("koinapy") is None:
        p.error("Koina prediction requires koinapy; use the configured Koina environment, --prosit-mgf, or --skip-prosit")
    os.environ["CUDA_VISIBLE_DEVICES"]=a.gpu
    st=Stages(a.out_dir,a.force); n=a.limit_groups or a.n_groups
    json_write(st.root/"run_manifest.json",dict(status="running",args=vars(a),paths={},versions=st.versions))
    if a.sample_dir:
        sample=Path(a.sample_dir).resolve(); expected,identities=validate_sample(sample,n,a.prefer_spectra)
        meta=json.loads((sample/"massTest_meta.json").read_text(encoding="utf8"))
        if meta["min_spectra"]!=a.min_spectra: raise ValueError("Sample min_spectra differs")
        # Imported sample dependencies are its files, not a pipeline stage manifest.
        sample_inputs=[sample/"massTest.mgf",sample/"massTest_groups.tsv",sample/"massTest_meta.json"]
        st.paths["sample"]=str(sample)
    else:
        if not a.train_mgf_xz: p.error("Supply --train-mgf-xz or --sample-dir")
        cfg=dict(n=n,seed=a.seed,min_spectra=a.min_spectra,prefer_spectra=a.prefer_spectra,max_spectra=a.max_spectra)
        def sample_action(d):
            opts=["--mgf-xz",Path(a.train_mgf_xz).resolve(),"--out-dir",d,"--n-groups",n,"--min-spectra",a.min_spectra,"--prefer-spectra",a.prefer_spectra,"--seed",a.seed,"--threads",a.threads]
            if a.max_spectra: opts += ["--max-spectra",a.max_spectra]
            st.command(HERE/"sample_massivekb_test.py",*opts)
        sample=st.run("sample",cfg,[a.train_mgf_xz],sample_action,lambda d:validate_sample(d,n,a.prefer_spectra))
        expected,identities=validate_sample(sample,n,a.prefer_spectra); sample_inputs=[sample]
    def publish(status):
        json_write(st.root/"run_manifest.json",dict(status=status,args=vars(a),paths=st.paths,versions=st.versions))
    publish("sample_complete")
    if a.stop_after=="sample": return
    real=sample/"massTest.mgf"
    def validate_filter(d):
        counts,ids=scan(d/"massTest_filtered.mgf",True)
        if counts!=expected or ids!=identities: raise ValueError("Filtering lost or changed spectrum identities")
    filt=st.run("filtered",dict(tol=a.tol_filter),sample_inputs,
                lambda d:st.command(FILTER,real,d/"massTest_filtered.mgf","--tol",a.tol_filter,"--workers",a.workers),validate_filter)
    qdir=None; qexpected={}
    if not a.skip_prosit:
        def predict(d):
            if a.prosit_mgf:
                shutil.copyfile(a.prosit_mgf,d/"prosit_predicted.mgf")
                got,_=scan(d/"prosit_predicted.mgf")
                with (d/"prosit_skipped.csv").open("w",encoding="utf-8",newline="") as fh:
                    w=csv.writer(fh);w.writerow(["sequence","charge","reason","detail"])
                    w.writerows((s,c,"external_missing","") for s,c in sorted(set(expected)-set(got)))
            else:
                keys=sorted(expected)
                np.savez(d/"massTest_seqs.npz",seqs=np.array([k[0] for k in keys]),charges=np.array([k[1] for k in keys]))
                st.command(PROSIT,"--npz",d/"massTest_seqs.npz","--out",d/"prosit_predicted.mgf","--cache",a.prosit_cache or st.root/"koina_cache.tsv.gz","--skipped-out",d/"prosit_skipped.csv")
        def validate_prediction(d):
            got,_=scan(d/"prosit_predicted.mgf")
            with (d/"prosit_skipped.csv").open(encoding="utf8") as fh: skipped=list(csv.DictReader(fh))
            absent={(r["sequence"],int(r["charge"])) for r in skipped}
            if any(n!=1 for n in got.values()) or not set(got)<=set(expected) or absent!=set(expected)-set(got) or len(absent)!=len(skipped):
                raise ValueError("Prosit success/skip partition differs from sampled groups")
        qdir=st.run("prosit",dict(cache=str(a.prosit_cache),mode="reanchor"),sample_inputs+([a.prosit_mgf] if a.prosit_mgf else []),predict,validate_prediction)
        qexpected,_=scan(qdir/"prosit_predicted.mgf")
    grouped={}
    for name,mgf,inputs,counts,pred in [("real",real,sample_inputs,expected,False),("filtered",filt/"massTest_filtered.mgf",[filt],expected,False)]+([("prosit",qdir/"prosit_predicted.mgf",[qdir],qexpected,True)] if qexpected else []):
        grouped[name]=st.run("grouped_"+name,{},inputs,
            lambda d,m=mgf,q=pred:st.command(HERE/"build_groups.py","--prediction-mgf" if q else "--original-mgf",m,"--out-dir",d),
            lambda d,c=counts,q=pred:validate_groups(d,c,q))
    for name in ("real","filtered"):
        def check_average(d):
            if {x.name for x in d.glob("*.mgf")}!={make_group_filename(s,c) for s,c in expected}: raise ValueError("Missing average files")
            for key in expected:
                counts,_=scan(d/make_group_filename(*key))
                if counts!={key:1}: raise ValueError("Invalid average spectrum")
        grouped["avg_"+name]=st.run("avg_"+name,dict(tol=a.tol_average),[grouped[name]],
            lambda d,g=grouped[name]:st.command(AVERAGE,"--mgf-dir",g,"--out-dir",d,"--tol",a.tol_average),check_average)
    publish("prepare_complete")
    if a.stop_after=="prepare": return
    metric_paths=[]
    for domain in domains:
        transformed={}; embeds={}
        for name,original in grouped.items():
            if domain=="none": mgfs=original
            else:
                def check_transform(d,g=original):
                    if {x.name for x in d.glob("*.mgf")}!={x.name for x in g.glob("*.mgf")}: raise ValueError("Transformed file set mismatch")
                    for path in g.glob("*.mgf"):
                        if scan(path)!=scan(d/path.name): raise ValueError("Transform changed group counts")
                mgfs=st.run(name+"_"+domain,dict(transform=domain),[original],
                    lambda d,g=original:st.command(HERE/"transform_mgf_intensity.py","--in",g,"--out",d,"--transform",domain),check_transform)
            transformed[name]=mgfs
            embeds[name]=st.run("emb_"+name+"_"+domain,dict(domain=domain,gpu=a.gpu),[mgfs,a.model],
                lambda d,g=mgfs:st.command(EMBED,"--mgf_dir",g,"--output_dir",d,"--model",Path(a.model).resolve(),"--strict-preprocessing"),
                lambda d,g=mgfs:validate_embeddings(d,g))
        def compare(d):
            opts=["--ref-grouped-dir",transformed["real"],"--src-grouped-dir",transformed["filtered"],"--ref-emb-dir",embeds["real"],"--src-emb-dir",embeds["filtered"],
                  "--avg-ref-grouped-dir",transformed["avg_real"],"--avg-src-grouped-dir",transformed["avg_filtered"],"--avg-ref-emb-dir",embeds["avg_real"],"--avg-src-emb-dir",embeds["avg_filtered"],"--domain",domain,"--out-dir",d]
            if "prosit" in embeds: opts += ["--prosit-grouped-dir",transformed["prosit"],"--prosit-emb-dir",embeds["prosit"]]
            st.command(HERE/"pairwise_cosine.py",*opts)
        def check_metrics(d):
            import pandas as pd
            df=pd.read_csv(d/"group_metrics.csv")
            if len(df)!=11*len(expected) or df.duplicated(["group_file","source","mode"]).any(): raise ValueError("Incomplete metrics")
        m=st.run("metrics_"+domain,dict(domain=domain),list(embeds.values())+list(transformed.values()),compare,check_metrics)
        metric_paths.append(m)
    result=st.run("summary",dict(domains=domains),metric_paths,
        lambda d:st.command(HERE/"summary_all.py","--out-dir",d,"--metrics",*[m/"group_metrics.csv" for m in metric_paths]),
        lambda d: (_ for _ in ()).throw(ValueError("Missing summary")) if not (d/"summary.csv").is_file() else None)
    publish("complete")
    print("COMPLETE",result/"summary.csv",flush=True)


if __name__ == "__main__":
    main()
