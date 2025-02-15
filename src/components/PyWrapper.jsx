import { useEffect, useState } from "react";
import { loadPyodide } from "pyodide";

function PyWrapper() {

    const [pyodideInstance, setPyodideInstance] = useState(null);
    const [code, setCode] = useState("1+1");
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        async function loadPy() {
            try {
                const pyodide = await loadPyodide({
                    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/"
                });
                await pyodide.loadPackage("micropip");
                const micropip = await pyodide.pyimport("micropip")
                await micropip.install("numpy")
                setPyodideInstance(pyodide);
            
            } catch (err) {
                setError(`Failed to load Pyodide ${err}`);
            }
        }
        loadPy();
    }, []);

    const runCode = async () => {
        if (!pyodideInstance) return;
        try {
            setError(null);
            const res = await pyodideInstance.runPythonAsync(code);
            setResult(res);
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
                style={{ width: '100%', marginBottom: '10px' }}
            />
            <button onClick={runCode}>Run Code</button>
            
            <div style={{ marginTop: '10px' }}>
                {error ? (
                    <p style={{ color: 'red' }}>Error: {error}</p>
                ) : result !== null ? (
                    <p>Output: {result}</p>
                ) : (
                    <p>Loading Pyodide...</p>
                )}
            </div>
        </div>
    );
}

export default PyWrapper;
