import json
def md(s): return {"cell_type":"markdown","metadata":{},"source":s.strip("\n").splitlines(True)}
def code(s): return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s.strip("\n").splitlines(True)}
cells=[
md("""
# 00 - Build Offline Wheels (CPU, free)

**Mode:** CPU session, Internet ON. **GPU cost: 0 h.**

**Why:** every GPU session that runs `pip install` bills you for the install time. We build the missing packages once here, publish them as the `llm-lab-wheels` dataset, and later GPU notebooks install them offline in seconds.

**Key idea:** Kaggle's image already has `torch` and friends. We do *not* download those (several GB, and replacing torch can break the image). A pip dry-run tells us exactly which packages would change, and we download only those.
"""),
code("""
import sys, platform, subprocess, json, importlib.metadata as md
print(sys.version); print(platform.platform())
for p in ["torch","transformers","trl","peft","accelerate","bitsandbytes","datasets","vllm","pydantic","pyyaml","psutil","tqdm"]:
    try: print(f"{p:14s}", md.version(p))
    except md.PackageNotFoundError: print(f"{p:14s}", "-- not installed")
"""),
md("## 1. Target pins\nThese are the versions locked in `uv.lock` for the local experiments (reproducibility, AGENTS.md rule 1)."),
code("""
PINS = ["transformers==5.16.1","trl==1.12.0","peft==0.20.0","accelerate==1.15.0",
        "bitsandbytes==0.50.2","datasets==5.0.1","pydantic==2.13.5","pyyaml==6.0.3",
        "psutil==7.2.2","tqdm==4.70.0"]
NEVER_DOWNLOAD = ("torch","nvidia-","triton","torchvision","torchaudio")  # owned by Kaggle's image
"""),
md("## 2. Dry-run: what would pip change?"),
code("""
def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

rc, out, err = sh([sys.executable,"-m","pip","install","--dry-run","--quiet","--report","/tmp/report.json",*PINS])
print("rc", rc); print(err[-2000:] if rc else "")
report = json.load(open("/tmp/report.json"))
todo = sorted((i["metadata"]["name"], i["metadata"]["version"]) for i in report["install"])
for n,v in todo: print(f"  {n}=={v}")
bad = [n for n,_ in todo if n.lower().startswith(NEVER_DOWNLOAD)]
assert not bad, f"pip wants to replace Kaggle-owned packages {bad}; stop and loosen pins"
"""),
md("## 3. Download only those wheels"),
code("""
import os, shutil
WHEELS = "/kaggle/working/wheels"
shutil.rmtree(WHEELS, ignore_errors=True); os.makedirs(WHEELS)
for n,v in todo:
    rc,_,err = sh([sys.executable,"-m","pip","download","--no-deps","--only-binary=:all:","-d",WHEELS,f"{n}=={v}"])
    if rc:  # sdist-only package: build a wheel instead
        rc,_,err = sh([sys.executable,"-m","pip","wheel","--no-deps","-w",WHEELS,f"{n}=={v}"])
    print("OK " if rc==0 else "FAIL", n, v, err[-300:] if rc else "")
files = sorted(os.listdir(WHEELS))
print(len(files), "files,", round(sum(os.path.getsize(f"{WHEELS}/{f}") for f in files)/1e6,1), "MB")
"""),
md("## 4. Verify: offline install into a clean target dir, then load the Qwen3.5 config"),
code("""
rc,out,err = sh([sys.executable,"-m","pip","install","--no-index","--find-links",WHEELS,"--target","/tmp/verify_site","--no-deps",*[f"{n}=={v}" for n,v in todo]])
print("offline install rc:", rc, err[-1000:] if rc else "")
check = '''
import sys; sys.path.insert(0, "/tmp/verify_site")
import transformers, trl, peft
print("transformers", transformers.__version__, "trl", trl.__version__, "peft", peft.__version__)
from transformers import AutoConfig
for m in ["Qwen/Qwen3.5-2B","Qwen/Qwen3.5-4B"]:
    print(m, "->", AutoConfig.from_pretrained(m, trust_remote_code=True).model_type)
from trl import GRPOConfig, GRPOTrainer, SFTTrainer
print("TRL GRPO/SFT import OK")
'''
rc,out,err = sh([sys.executable,"-c",check]); print(out); print(err[-1500:] if rc else "")
"""),
md("## 5. (Optional) vLLM impact report - report only, nothing downloaded\nvLLM pins its own `torch`; installing it on top of Kaggle's image may break everything else. Decide at NB 06 using this report."),
code("""
rc,out,err = sh([sys.executable,"-m","pip","install","--dry-run","--quiet","--report","/tmp/vllm.json","vllm"])
if rc: print(err[-1500:])
else:
    r = json.load(open("/tmp/vllm.json"))["install"]
    names = sorted((i["metadata"]["name"], i["metadata"]["version"]) for i in r)
    print(len(names), "packages would change; torch-family:", [x for x in names if x[0].lower().startswith(NEVER_DOWNLOAD)])
    json.dump(names, open("/kaggle/working/vllm_impact.json","w"), indent=1)
"""),
md("## 6. Manifest"),
code("""
json.dump({"python": sys.version, "pins": PINS, "wheels": files}, open(f"{WHEELS}/MANIFEST.json","w"), indent=1)
print(open(f"{WHEELS}/MANIFEST.json").read()[:1500])
"""),
md("""
## Next (manual)
1. Check the verify cell printed `TRL GRPO/SFT import OK` and both Qwen model types.
2. Open this notebook's **Output** tab -> **New Dataset** -> select the `wheels` folder -> title `llm-lab-wheels` -> **Public** (public data doesn't use your 214 GB private quota).
"""),
]
nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
json.dump(nb,open("kaggle/kernels/00-build-wheels/00_build_wheels.ipynb","w"),indent=1)
json.dump({"id":"ycheng22/00-build-wheels","title":"00 Build Wheels - Local LLM Lab","code_file":"00_build_wheels.ipynb","language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":False,"enable_tpu":False,"enable_internet":True,"dataset_sources":[],"competition_sources":[],"kernel_sources":[],"model_sources":[]},open("kaggle/kernels/00-build-wheels/kernel-metadata.json","w"),indent=2)
