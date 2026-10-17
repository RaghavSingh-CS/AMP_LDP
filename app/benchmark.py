"""Read-only, fresh-process demo measurements; no research inference or writes.

Use --offline to test with external Python network connections blocked. The
browser's static assets are separately inspected in the live UI validation.
"""

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
from time import perf_counter, sleep
from urllib.request import urlopen

from app.config import REPOSITORY_ROOT as ROOT


def measure(offline=False):
    environment = os.environ.copy()
    if offline:
        environment["PYTHONPATH"] = str(ROOT / "tests/offline") + os.pathsep + str(ROOT)
    with socket.socket() as handle:
        handle.bind(("127.0.0.1", 0))
        port = handle.getsockname()[1]
    command = [sys.executable, "-m", "streamlit", "run", "app/main.py", "--server.headless", "true",
               "--server.address", "127.0.0.1", "--server.port", str(port), "--browser.gatherUsageStats", "false"]
    started = perf_counter()
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        while perf_counter() - started < 30:
            if process.poll() is not None:
                raise RuntimeError("Fresh Streamlit process exited before health check.")
            try:
                with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=1) as response:
                    if response.status == 200:
                        health_seconds = perf_counter() - started
                        break
            except OSError:
                sleep(.1)
        else:
            raise RuntimeError("Fresh Streamlit health check timed out.")
        with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
            root_status = response.status
    finally:
        process.terminate()
        process.wait(timeout=10)

    # Separate fresh process includes UI-library imports and any first-page cache fill.
    program = '''
import json
from time import perf_counter
started=perf_counter()
from streamlit.testing.v1 import AppTest
app=AppTest.from_file("app/main.py",default_timeout=30).run()
timings={"overview_fresh_process_including_imports":perf_counter()-started}
def run(name,action):
 t=perf_counter();action();timings[name]=perf_counter()-t
 assert not app.exception,str(app.exception)
def button(label):return next(b for b in app.button if b.label==label)
for page,key in [("Customer Risk Explorer","explorer"),("Model Performance","performance"),("Demo Guide","guide")]:
 run(key+"_first",lambda:app.radio[0].set_value(page).run())
 run(key+"_repeat",lambda:app.run())
app.selectbox[0].set_value("high_theft").run()
run("inspector_first",lambda:button("Inspect verified example").click().run())
run("inspector_repeat",lambda:app.run())
run("inspector_next_rank",lambda:button("Next rank").click().run())
run("overview_return",lambda:app.radio[0].set_value("Overview").run())
print(json.dumps(timings))
'''
    child = subprocess.run([sys.executable, "-c", program], cwd=ROOT, env=environment,
                           text=True, capture_output=True, timeout=120)
    if child.returncode:
        raise RuntimeError(child.stderr[-4000:])
    timings = json.loads(child.stdout.strip())
    return {"machine": {"platform": platform.platform(), "python": platform.python_version(), "cpu": platform.processor(),
                        "logical_cpus": os.cpu_count()},
            "network_profile": "Python outbound blocked; loopback only" if offline else "normal local process",
            "startup_to_health_seconds": round(health_seconds, 4), "root_http_status": root_status,
            "page_seconds": {key: round(value, 4) for key, value in timings.items()},
            "interpretation": "Single local run; fresh Python/Streamlit processes with OS disk cache uncontrolled. AppTest times include rendering/serialization, not browser paint. No before/after improvement claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    print(json.dumps(measure(args.offline), indent=2))


if __name__ == "__main__":
    main()
