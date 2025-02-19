import { useEffect, useState } from "react";
import { loadPyodide } from "pyodide";
import Terminal from "./Terminal";

function PyWrapper() {

    const [pyodideInstance, setPyodideInstance] = useState(null);
    const [code, setCode] = useState("import numpy as np\nnp.eye(3)");
    const [result, setResult] = useState([]);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        async function loadPy() {
            try {
                setIsLoading(true);
                const pyodide = await loadPyodide({
                    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/"
                });
                
                pyodide.globals.set('_stdout_callback', (text) => {
                    setResult(prev => [...prev, text]);
                });
                
                pyodide.runPython(`
                    import sys
                    from pyodide.ffi import create_proxy
                    from io import StringIO
                    
                    class WebConsole(StringIO):
                        def write(self, text):
                            _stdout_callback(text)
                            return len(text)
                            
                    sys.stdout = WebConsole()
                `);
                
                await pyodide.loadPackage("micropip");
                const micropip = await pyodide.pyimport("micropip")
                await micropip.install("numpy")
                setPyodideInstance(pyodide);
                setIsLoading(false);
            
            } catch (err) {
                setError(`Failed to load Pyodide ${err}`);
                setIsLoading(false);
            }
        }
        loadPy();
    }, []);

    const runCode = async () => {
        if (!pyodideInstance) return;
        try {

            setError(null);
            setResult([]);
            
            const res = await pyodideInstance.runPythonAsync(code);
            if (res !== undefined) {
                console.log(res.toJs())
                setResult(prev => [...prev, res.toJs()]);
            }
        } catch (err) {
            setError(err.message);
        }
    };

    return (
        <div>
            <textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="Enter Python code here"
                rows={5}
                style={{ 
                    width: '100%', 
                    marginBottom: '10px',
                    fontFamily: 'monospace',
                    whiteSpace: 'pre',
                    tabSize: 4,
                    MozTabSize: 4
                }}
            />
            <button onClick={runCode} disabled={isLoading || !pyodideInstance}>
                Run Code
            </button>
            
            <Terminal 
                isLoading={isLoading}
                error={error}
                result={result}
            />
        </div>
    );
}

export default PyWrapper;
