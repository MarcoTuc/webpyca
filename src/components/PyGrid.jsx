import { useEffect, useState, useCallback } from "react";
import { loadPyodide } from "pyodide";
import P5Wrapper from "./P5Wrapper";

const initialCode = `
import numpy as np

size = 150

np.random.rand(size, size)

`


function sketch(p) {
    let currentCells = [];
    const canvasWidth = 600;
    const canvasHeight = 600;
    const cellSize = 1;

    p.setup = function() {
        p.createCanvas(canvasWidth, canvasHeight, p.P2D);
        p.noLoop();
        p.pixelDensity(1);
    }

    p.draw = function() {
        p.background(255);
        if (!Array.isArray(currentCells)) return;
        
        const rowCount = currentCells.length;
        if (rowCount === 0) return;
        const colCount = currentCells[0].length;

        p.loadPixels();
        
        for (let i = 0; i < rowCount; i++) {
            for (let j = 0; j < colCount; j++) {
                const value = currentCells[i][j];
                const color = Math.floor(255 * (1 - value));
                
                const x = j * cellSize;
                const y = i * cellSize;
                
                for (let dy = 0; dy < cellSize; dy++) {
                    for (let dx = 0; dx < cellSize; dx++) {
                        const idx = 4 * ((y + dy) * canvasWidth + (x + dx));
                        p.pixels[idx] = color;     // R
                        p.pixels[idx + 1] = color; // G
                        p.pixels[idx + 2] = color; // B
                        p.pixels[idx + 3] = 255;   // A
                    }
                }
            }
        }
        
        p.updatePixels();
    }

    p.updateCells = function(newCells) {
        const start = performance.now();
        currentCells = newCells;
        p.redraw();
        console.log(`P5 redraw time: ${(performance.now() - start).toFixed(2)}ms`);
        console.log(`Projected FPS : ${(1000/(performance.now() - start)).toFixed(2)} FPS`);
    }

    return p;
}

function PyGrid() {
    
    const [pyodideInstance, setPyodideInstance] = useState(null);
    const [code, setCode] = useState(initialCode);
    const [error, setError] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [p5Instance, setP5Instance] = useState(null);

    const initializeSketch = useCallback((p) => {
        const instance = sketch(p);
        setP5Instance(instance);
        return instance;
    }, []);

    // Load Pyodide once on component mount
    useEffect(() => {

        let mounted = true;

        async function loadPy() {
            try {
                const pyodide = await loadPyodide({
                    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.27.2/full/"
                });
                
                await pyodide.loadPackage("micropip");
                const micropip = await pyodide.pyimport("micropip");
                await micropip.install("numpy");
                
                if (mounted) {
                    setPyodideInstance(pyodide);
                    setIsLoading(false);
                }
            } catch (err) {
                if (mounted) {
                    setError(`Failed to load Pyodide: ${err}`);
                    setIsLoading(false);
                }
            }
        }

        loadPy();
        return () => { mounted = false; };
    }, []);

    const runCode = async () => {
        if (!pyodideInstance || !p5Instance) return;
        
        const timings = {};
        const mark = (label) => {
            timings[label] = performance.now();
        };
        const measure = (start, end) => {
            return `${(timings[end] - timings[start]).toFixed(2)}ms`;
        };

        try {
            setError(null);
            mark('start');
            
            mark('pythonExecution');
            const result = await pyodideInstance.runPythonAsync(code);
            mark('pythonDone');
            
            mark('conversion');
            const jsResult = result.toJs();
            mark('conversionDone');
            
            if (Array.isArray(jsResult)) {
                mark('renderStart');
                p5Instance.updateCells(jsResult);
                mark('renderDone');
                
                // Log all timings
                console.log({
                    'Python Execution': measure('pythonExecution', 'pythonDone'),
                    'Array Conversion': measure('conversion', 'conversionDone'),
                    'Rendering': measure('renderStart', 'renderDone'),
                    'Total Time': measure('start', 'renderDone')
                });
            } else {
                setError("Not an array");
            }
        } catch (err) {
            setError(err.message);
        }
    };

    return (
        <div className="ascii-play-container">
            <div className="left-panel">
                <div className="editor-section">
                    <textarea
                        className="code-editor"
                        value={code}
                        onChange={(e) => setCode(e.target.value)}
                        placeholder="Enter Python code here"
                        rows={20}
                    />
                    <div className="controls">
                        <button 
                            className="run-button"
                            onClick={runCode} 
                            disabled={isLoading || !pyodideInstance}
                        >
                            {isLoading ? <span style={{color: '#ff6b6b'}}>Loading</span> : 'Run'}
                        </button>
                        {error && <div className="error-message">{error}</div>}
                    </div>
                </div>
            </div>
            <div className="right-panel">
                <div className="output-section">
                    <div className="panel-header">Output</div>
                    <P5Wrapper 
                        sketch={initializeSketch}
                        id="pygrid-container"
                    />
                </div>
            </div>
        </div>
    );
}

export default PyGrid;
