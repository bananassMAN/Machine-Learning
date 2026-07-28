import json
import io
import base64
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from IPython.display import display

with open("skill_lab1.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global execution context
exec_globals = {
    "display": lambda x: print(x.to_string() if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series) else x)
}

for cell_idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    
    source_code = "".join(cell["source"])
    cell["execution_count"] = cell_idx + 1
    outputs = []
    
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    plt.close('all')
    
    try:
        # Split lines to evaluate trailing expressions (like in Jupyter)
        lines = [l for l in source_code.split('\n') if l.strip() and not l.strip().startswith('#')]
        
        if lines:
            # Execute all except last line
            code_exec = "\n".join(source_code.split('\n')[:-1]) if len(lines) > 1 else ""
            last_line = lines[-1]
            
            # Try to compile code
            try:
                # If last line is an expression, evaluate it
                code_ast = compile(source_code, '<string>', 'exec')
                exec(code_ast, exec_globals)
            except Exception:
                exec(source_code, exec_globals)
        else:
            exec(source_code, exec_globals)
        
        stdout_val = sys.stdout.getvalue()
        if stdout_val:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": stdout_val.splitlines(True)
            })
            
        fig_nums = plt.get_fignums()
        for fig_num in fig_nums:
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            outputs.append({
                "data": {
                    "image/png": img_str + "\n",
                    "text/plain": "<Figure size ... with ... Axes>"
                },
                "metadata": {},
                "output_type": "display_data"
            })
            plt.close(fig)
            
    except Exception as e:
        print(f"Error in cell {cell_idx+1}: {e}")
        stdout_val = sys.stdout.getvalue()
        if stdout_val:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": stdout_val.splitlines(True)
            })
        outputs.append({
            "ename": type(e).__name__,
            "evalue": str(e),
            "output_type": "error",
            "traceback": [str(e)]
        })
    finally:
        sys.stdout = old_stdout
        
    cell["outputs"] = outputs

with open("skill_lab1.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook execution complete.")
