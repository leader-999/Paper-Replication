"""Run the reviewed author Section 4 notebook without changing its calculations."""
from pathlib import Path
import json
import os
import platform
from importlib.metadata import version

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
notebook = json.loads((ROOT / "E2E_Section4.ipynb").read_text(encoding="utf-8"))
namespace = {"__name__": "__main__"}
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell.get("source", []))
    if source.strip():
        print(f"Running author cell {index}", flush=True)
        exec(compile(source, f"E2E_Section4.ipynb:cell-{index}", "exec"), namespace)
outputs = ROOT / "outputs"
outputs.mkdir(exist_ok=True)
namespace["df_aggcom"].to_csv(outputs / "section4_original_costs.csv", encoding="utf-8-sig")
versions = {"python": platform.python_version()}
for name in ("numpy", "pandas", "scipy", "seaborn", "tqdm"):
    versions[name] = version(name)
(outputs / "section4_environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")
print("\nRESULTS (author code, disclosed subset):")
print(namespace["df_aggcom"].round(2).to_string())
print("Input rows:", len(__import__("pandas").read_csv(ROOT / "df_test.csv")))
print("Unique SKU groups:", len(namespace["df_test"]))
