import { useEffect, useState, useCallback } from "react";
import { loadPyodide } from "pyodide";
import P5Wrapper from "./components/P5Wrapper";

import AceEditor from "react-ace";
import CodeMirror from "@uiw/react-codemirror"
import { python } from '@codemirror/lang-python'
import { okaidia } from '@uiw/codemirror-theme-okaidia';
import { atomone } from '@uiw/codemirror-theme-atomone'

const initialImports =  
`
import numpy as np
import time

`
const initialCode = `
size = 150
np.random.rand(size, size)

`

function sketch(p, config) {
    
    let currentCells = [];
    const canvasWidth = config.width;
    const canvasHeight = config.height;
    const cellSize = config.cellSize;

    p.setup = function() {
        p.createCanvas(canvasWidth, canvasHeight, p.P2D);
        p.noLoop();
        p.pixelDensity(1);
        // p.noSmooth();
    }

    p.draw = function() {
        p.background('#1e1e1e');
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
        // console.log(`P5 redraw time: ${(performance.now() - start).toFixed(2)}ms`);
        // console.log(`Projected FPS : ${(1000/(performance.now() - start)).toFixed(2)} FPS`);
    }

    return p;
}

function PyGrid() {
    
    const [pyodideInstance, setPyodideInstance] = useState(null);
    const [p5Instance, setP5Instance] = useState(null);
    
    const [isLoading, setIsLoading] = useState(true);
    const [isRunning, setIsRunning] = useState(false);
    
    const [imports, setImports] = useState(initialImports);
    const [code, setCode] = useState(initialCode);
    
    const [codeChanged, setCodeChanged] = useState(true);
    const [error, setError] = useState(null);

    const [canvasConfig, setCanvasConfig] = useState({
        width: 600,
        height: 600,
        cellSize: 1
    });

    const initializeSketch = useCallback((p) => {
        const instance = sketch(p, canvasConfig);
        setP5Instance(instance);
        return instance;
    }, [canvasConfig]);

    function codeUpdater(newCode) {
        console.log("Code changed")
        setCode(newCode)
        setCodeChanged(true)
        console.log(`new flag ${codeChanged}`)
    }

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
                // import standard packages when initializing the engine
                await pyodide.runPython(imports)
                
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

    useEffect(() => {
        let animationFrameId;

        const loop = () => {
            if (isRunning) {
                runCode();
                animationFrameId = requestAnimationFrame(loop);
            }
        };

        if (isRunning) {
            loop();
        }

        return () => {
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
            }
        };
    }, [isRunning]);

    const runCode = async () => {
        if (!pyodideInstance || !p5Instance) return;
        
        try {
            setError(null);
            
            const result = await pyodideInstance.runPythonAsync(code);
            const jsResult = result.toJs();
            
            if (Array.isArray(jsResult)) {       
                if (codeChanged) {
                    // Temporarily stop the animation
                    setIsRunning(false);
                    
                    const receivedSize = jsResult.length;
                    const newSize = canvasConfig.width / receivedSize;
                    setCanvasConfig(prev => ({
                        ...prev,
                        cellSize: newSize
                    }));
                    
                    // Wait for next tick to ensure canvas is updated
                    setTimeout(() => {
                        setCodeChanged(false);
                        setIsRunning(true);
                    }, 0);
                    
                }

                p5Instance.updateCells(jsResult);
            } else {
                setError("Not an array");
            }
        } catch (err) {
            setError(err.message);
        }
    };

    // const handleKeyDown = (e) => {
    //     if (e.key === 'Tab') {
    //         e.preventDefault();
    //         const cursorPosition = e.target.selectionStart;
    //         const newText = code.slice(0, cursorPosition) + '    ' + code.slice(cursorPosition);
    //         setCode(newText);
            
    //         // Move cursor after the inserted tab
    //         // e.target.setSelectionRange(cursorPosition + 4, cursorPosition + 4);
    //     }
    // };  

    return (
        <div className="ascii-play-container">
            <div className="left-panel">
                <div className="editor-section">
                    <CodeMirror
                        value={code}
                        onChange={codeUpdater}
                        theme={atomone}
                        extensions={[python()]}
                    />
                    <div className="controls-container">
                        <button 
                            className="run-button"
                            onClick={() => setIsRunning(!isRunning)}
                        >
                            {isRunning ? 'Stop' : 'Run'}
                        </button>
                    </div>
                    {error && <div className="error-message">{error}</div>}
                </div>
            </div>
            <div className="right-panel">
                <div className="output-section">
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
