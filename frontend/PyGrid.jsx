import { useEffect, useState, useCallback } from "react";
import { loadPyodide } from "pyodide";

import CodeMirror from "@uiw/react-codemirror"
import { python } from '@codemirror/lang-python'

import P5Wrapper from "./components/P5Wrapper";
import { useTheme } from './components/ThemeProvider';
import ThemeToggle from "./components/ThemeToggle";
import RunButton from "./components/RunButton";
import FPSInput from "./components/FPSInput";
import saved_automata from "./SavedAutomata";
import AutomatonSelect from "./components/AutomatonSelect";
import ReinitializeButton from "./components/ReinitializeButton";

const initialImports =  
`
import numpy as np
import time

`
const initialCode = saved_automata.lenia

function sketch(p, config, pyodideInstance, theme, isRGB) {
    
    let grid = {
        cells: [],
        x: 0,
        y: 0,
        scale: 1,
        cellSize: config.cellSize,
        width: config.width,
        height: config.height
    };

    // Add these variables to track dragging
    let isDragging = false;
    let lastMouseX = 0;
    let lastMouseY = 0;

    p.setup = function() {
        p.createCanvas(config.width, config.height, p.P2D);
        p.noLoop();
        p.pixelDensity(1);
        p.noSmooth();
    }

    p.updateConfig = function(newConfig) {
        grid.width = newConfig.width;
        grid.height = newConfig.height;
        grid.cellSize = newConfig.cellSize;
    }

    p.mouseDragged = function() {
        if (p.keyIsDown(p.CONTROL)) {
            // Handle dragging
            const dx = p.mouseX - lastMouseX;
            const dy = p.mouseY - lastMouseY;
            grid.x += dx;
            grid.y += dy;
            lastMouseX = p.mouseX;
            lastMouseY = p.mouseY;
            p.redraw();
        } else {
            handleMouse()
        }
    }

    p.mousePressed = function() {
        if (p.keyIsDown(p.CONTROL)) {
            isDragging = true;
            lastMouseX = p.mouseX;
            lastMouseY = p.mouseY;
        } else {
            handleMouse()
        }
    }

    p.mouseReleased = function() {
        isDragging = false;
    }

    p.mouseWheel = function(e) {
        // Get canvas element
        const canvas = p.canvas;
        const rect = canvas.getBoundingClientRect();
        
        // Check if mouse is over the canvas
        const isOverCanvas = 
            e.clientX >= rect.left && 
            e.clientX <= rect.right && 
            e.clientY >= rect.top && 
            e.clientY <= rect.bottom;

        if (isOverCanvas) {
            const s = e.delta < 0 ? 1.05 : 0.95;
            grid.scale *= s;
            grid.x = p.mouseX * (1-s) + grid.x * s;
            grid.y = p.mouseY * (1-s) + grid.y * s;
            p.redraw();
            return false; // Prevent default only when over canvas
        }
    }

    function handleMouse() {
        if (p.mouseX >= 0 && p.mouseX < grid.width && 
            p.mouseY >= 0 && p.mouseY < grid.height) {
            // Convert mouse coordinates to grid coordinates accounting for transformation
            const mouseX = Math.floor((p.mouseX - grid.x) / (grid.cellSize * grid.scale));
            const mouseY = Math.floor((p.mouseY - grid.y) / (grid.cellSize * grid.scale));
            
            if (pyodideInstance) {
                try {
                    pyodideInstance.runPython(`
                        if 'spray' in globals() and callable(spray):
                            spray(${mouseY}, ${mouseX})
                    `);
                } catch (err) {
                    console.error("Failed to call spray:", err);
                }
            }
        }
    }

    p.updateTheme = function(newTheme) {
        theme = newTheme
    }

    p.updateRGB = function(newRGB) {
        isRGB = newRGB
    }

    p.draw = function() {

        const lightBg = { r: 0xF8, g: 0xF6, b: 0xF1 };  // #F8F6F1
        const darkBg = { r: 0x1E, g: 0x1E, b: 0x1E };  // #1E1E1E

        if (!Array.isArray(grid.cells)) return;
        const rowCount = grid.cells.length;
        if (rowCount === 0) return;
        const colCount = grid.cells[0].length;

        // Clear the entire canvas with the appropriate background color
        p.loadPixels();
        const bg = theme === 'dark' ? darkBg : lightBg;
        for (let i = 0; i < p.pixels.length; i += 4) {
            p.pixels[i] = bg.r;        // R
            p.pixels[i + 1] = bg.g;    // G
            p.pixels[i + 2] = bg.b;    // B
            p.pixels[i + 3] = 255;     // A
        }
        
        // Calculate the visible region of the grid
        const scaledCellSize = grid.cellSize * grid.scale;
        const startX = Math.max(0, Math.floor(-grid.x / scaledCellSize));
        const startY = Math.max(0, Math.floor(-grid.y / scaledCellSize));
        const endX = Math.min(colCount, Math.ceil((grid.width - grid.x) / scaledCellSize));
        const endY = Math.min(rowCount, Math.ceil((grid.height - grid.y) / scaledCellSize));


        for (let i = startY; i < endY; i++) {
            for (let j = startX; j < endX; j++) {        
                
                const cell = grid.cells[i][j];

                // Get base RGB values
                const baser = Math.floor(isRGB ? cell[0] * 255 : cell * 255);
                const baseg = Math.floor(isRGB ? cell[1] * 255 : cell * 255);
                const baseb = Math.floor(isRGB ? cell[2] * 255 : cell * 255);

                let r, g, b;
                if (theme === 'light') {
                    // In light theme, pinch white to lightBg
                    r = Math.min(255 - baser, lightBg.r);
                    g = Math.min(255 - baseg, lightBg.g);
                    b = Math.min(255 - baseb, lightBg.b);
                } else {
                    // In dark theme, pinch black to darkBg
                    r = Math.max(baser, darkBg.r);
                    g = Math.max(baseg, darkBg.g);
                    b = Math.max(baseb, darkBg.b);
                }
               
                // Calculate screen coordinates
                const screenX = Math.floor(j * scaledCellSize + grid.x);
                const screenY = Math.floor(i * scaledCellSize + grid.y);
                
                // Draw scaled cell
                const cellWidth = Math.ceil(scaledCellSize);
                const cellHeight = Math.ceil(scaledCellSize);
                
                for (let dy = 0; dy < cellHeight; dy++) {
                    const pixelY = screenY + dy;
                    if (pixelY < 0 || pixelY >= grid.height) continue;
                    
                    for (let dx = 0; dx < cellWidth; dx++) {
                        const pixelX = screenX + dx;
                        if (pixelX < 0 || pixelX >= grid.width) continue;
                        
                        const idx = 4 * (pixelY * grid.width + pixelX);
                        p.pixels[idx] = r;     // R
                        p.pixels[idx + 1] = g; // G
                        p.pixels[idx + 2] = b; // B
                        p.pixels[idx + 3] = 255;   // A
                    }
                }
            }
        }
        
        p.updatePixels();
    }

    p.updateCells = function(newCells) {
        grid.cells = newCells;
        p.redraw();
    }

    return p;
}

function PyGrid({ themes }) {

    const { theme } = useTheme();
    const [pyodideInstance, setPyodideInstance] = useState(null);
    const [p5Instance, setP5Instance] = useState(null);
    
    const [isLoading, setIsLoading] = useState(true);
    const [isRunning, setIsRunning] = useState(false);
    
    const [imports, setImports] = useState(initialImports);
    const [code, setCode] = useState(initialCode);
    
    const [error, setError] = useState(null);
    const [automatonType, setAutomatonType] = useState('lenia');

    const [canvasConfig, setCanvasConfig] = useState({
        width: 600,
        height: 600,
        cellSize: 1
    });

    const [storedCode, setStoredCode] = useState("");

    const [FPS, setFPS] = useState(30)
    const [isRGB, setIsRGB] = useState(false)

    const initializeSketch = useCallback((p) => {
        const instance = sketch(p, canvasConfig, pyodideInstance, theme, isRGB);
        setP5Instance(instance);
        return instance;
    }, [pyodideInstance, canvasConfig]);

    
    useEffect(() => {
        if (p5Instance) {
            p5Instance.updateTheme(theme)
            p5Instance.redraw()
        }
    }, [theme])

    useEffect(() => {
        if (p5Instance) {
            p5Instance.updateRGB(isRGB)
        }
    }, [isRGB])

    function codeUpdater(newCode) {
        setCode(newCode);
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
                await pyodide.runPython(`
                    import micropip
                    micropip.install("numpy")
                    micropip.install("scipy")
                    `)
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
        let intervalId;
        
        if (isRunning) {
            intervalId = setInterval(async () => {
                await Promise.resolve(runCode());
            }, 1000/FPS); // 2 FPS
        }

        return () => {
            if (intervalId) {
                clearInterval(intervalId);
            }
        };
    }, [isRunning, FPS]);

    async function checkCode(code) {
        const hasMainFunction = await pyodideInstance.runPythonAsync(`
            'main' in globals() and callable(globals()['main'])
        `);
        
        if (!hasMainFunction) {
            setError("No main() function defined");
            return false;
        }

        // Run one step to get dimensions
        const result = await pyodideInstance.runPythonAsync("main()");
        const jsResult = result.toJs();
        
        if (Array.isArray(jsResult)) {   
            const receivedSize = jsResult.length;
            const newSize = canvasConfig.width / receivedSize;
            const newConfig = {
                ...canvasConfig,
                cellSize: newSize
            };
            
            setCanvasConfig(newConfig);
            p5Instance.updateConfig(newConfig);

            if (typeof jsResult[0][0] === 'number') {
                setIsRGB(false)
            } else {
                setIsRGB(true)
            }
            
            // Store the new code and reinitialize
            setStoredCode(code);
            return jsResult; // Return the result directly
        }
        return false;
    }

    const runCode = async () => {
        if (!pyodideInstance || !p5Instance) return;
        
        try {
            // setError(null);
            
            // // Check if code has changed from stored version
            // if (code !== storedCode) {
            //     await pyodideInstance.runPythonAsync(code);
                
            //     const hasMainFunction = await pyodideInstance.runPythonAsync(`
            //         'main' in globals() and callable(globals()['main'])
            //     `);
                
            //     if (!hasMainFunction) {
            //         setError("No main() function defined");
            //         return;
            //     }

            //     // Run one step to get dimensions
            //     const result = await pyodideInstance.runPythonAsync("main()");
            //     const jsResult = result.toJs();
                
            //     if (Array.isArray(jsResult)) {   
                     
            //         const receivedSize = jsResult.length;
            //         const newSize = canvasConfig.width / receivedSize;
            //         const newConfig = {
            //             ...canvasConfig,
            //             cellSize: newSize
            //         };

            //         setCanvasConfig(newConfig);
            //         p5Instance.updateConfig(newConfig);

            //         if (typeof jsResult[0][0] === 'number') {
            //             setIsRGB(false)
            //         } else {
            //             setIsRGB(true)
            //         }
                    
            //         // Store the new code and reinitialize
            //         setStoredCode(code);
            //         await pyodideInstance.runPythonAsync(code);
            //     } else {
            //         setError("main() must return an array");
            //         return;
            //     }
            // }
            
            // Timing measurements
            const pythonStart = performance.now();
            
            // Convert the numpy array to a contiguous float32 array in Python
            await pyodideInstance.runPythonAsync(`
                _result = main()
                if isinstance(_result, np.ndarray):
                    _result = _result.astype(np.float32).ravel()
                else:
                    _result = np.array(_result, dtype=np.float32).ravel()
            `);
            
            // Get the flattened array directly as a TypedArray
            const flatArray = pyodideInstance.globals.get('_result').toJs();
            const pythonEnd = performance.now();
            
            const conversionStart = performance.now();
            // Reshape the array back to the original dimensions
            const size = Math.sqrt(flatArray.length / (isRGB ? 3 : 1));
            const jsResult = [];
            
            if (isRGB) {
                for (let i = 0; i < size; i++) {
                    const row = [];
                    for (let j = 0; j < size; j++) {
                        const idx = (i * size + j) * 3;
                        row.push([
                            flatArray[idx],
                            flatArray[idx + 1],
                            flatArray[idx + 2]
                        ]);
                    }
                    jsResult.push(row);
                }
            } else {
                for (let i = 0; i < size; i++) {
                    const row = [];
                    for (let j = 0; j < size; j++) {
                        row.push(flatArray[i * size + j]);
                    }
                    jsResult.push(row);
                }
            }
            const conversionEnd = performance.now();
            
            // Clean up the temporary Python variable
            pyodideInstance.runPython('del _result');
            
            const renderStart = performance.now();
            p5Instance.updateCells(jsResult);
            const renderEnd = performance.now();
            
            // Log timing results
            console.log(`---------------------------------------------`);
            console.log(`Python computation: ${(pythonEnd - pythonStart).toFixed(2)}ms`);
            console.log(`Array conversion: ${(conversionEnd - conversionStart).toFixed(2)}ms`);
            console.log(`Rendering: ${(renderEnd - renderStart).toFixed(2)}ms`);
            console.log(`Total: ${(renderEnd - pythonStart).toFixed(2)}ms`);
            
        } catch (err) {
            setError(err.message);
            setIsRunning(false);
        }
    };

    const handleRunClick = () => {
        if (!isRunning) {
            runCode().then(() => {
                if (!error) setIsRunning(true);
            });
        } else {
            setIsRunning(false);
        }
    };

    const handleReinitialize = async () => {
        const wasRunning = isRunning;
        setIsRunning(false); // Stop any running intervals
        setError(null);
        
        try {
            // Reset pyodide state by re-evaluating the imports and code
            await pyodideInstance.runPythonAsync(imports);
            await pyodideInstance.runPythonAsync(code);
            
            // Check if the main function exists
            const hasMainFunction = await pyodideInstance.runPythonAsync(`
                'main' in globals() and callable(globals()['main'])
            `);
            
            if (!hasMainFunction) {
                setError("No main() function defined");
                return;
            }
            
            // Run main() once to get the initial state and detect changes needed
            const result = await pyodideInstance.runPythonAsync("main()");
            const jsResult = result.toJs();
            
            if (!Array.isArray(jsResult)) {
                setError("main() must return an array");
                return;
            }
            
            // Update canvas configuration directly if code has changed
            if (code !== storedCode) {
                const receivedSize = jsResult.length;
                const newSize = canvasConfig.width / receivedSize;
                const newConfig = {
                    ...canvasConfig,
                    cellSize: newSize
                };
                
                // Update configuration before rendering
                setCanvasConfig(newConfig);
                p5Instance.updateConfig(newConfig);
                
                // Detect RGB mode
                const isRGBMode = typeof jsResult[0][0] !== 'number';
                setIsRGB(isRGBMode);
                p5Instance.updateRGB(isRGBMode);
                
                // Store the code for future reference
                setStoredCode(code);
            }
            
            // Reset any internal state in pyodide if needed
            await pyodideInstance.runPythonAsync(`
                # Reset any global variables that might be maintaining state
                if 'reset' in globals() and callable(globals()['reset']):
                    reset()
            `);
            
            // Get a fresh state after reset
            const freshResult = await pyodideInstance.runPythonAsync("main()");
            const freshState = freshResult.toJs();
            
            // Update the p5 instance with the fresh state
            p5Instance.updateCells(freshState);
            
            // Simulate a single step of the simulation after a short delay
            setTimeout(async () => {
                if (pyodideInstance) {
                    try {
                        const stepResult = await pyodideInstance.runPythonAsync("main()");
                        p5Instance.updateCells(stepResult.toJs());
                        
                        // Restore the previous running state if it was running
                        if (wasRunning) {
                            setIsRunning(true);
                        }
                    } catch (stepErr) {
                        console.error("Error in simulation step:", stepErr);
                    }
                }
            }, 50);
            
        } catch (err) {
            setError(err.message);
        }
    };

    function clearCanvas() {
        const size = Math.floor(canvasConfig.width / canvasConfig.cellSize);
        const emptyGrid = Array(size).fill().map(() => 
            isRGB 
                ? Array(size).fill().map(() => [0, 0, 0]) 
                : Array(size).fill(0)
        );
        
        p5Instance.updateCells(emptyGrid);
    };

    useEffect(() => {
        setCode(saved_automata[automatonType]);
        setIsRunning(false);
        if (p5Instance) {
            clearCanvas();
        }
    }, [automatonType]);

    return (
        <div className="ascii-play-container">
            
            <div className="right-panel">
                <div className="output-section">
                    <P5Wrapper 
                        sketch={initializeSketch}
                        id="pygrid-container"
                    />
                </div>
            </div>

            <div className="left-panel">
                <div className="editor-section">
                    <CodeMirror
                        className="codemirror-editor"
                        value={code}
                        onChange={codeUpdater}
                        theme={themes[theme]}
                        extensions={[python()]}
                        basicSetup={{
                            lineNumbers: false,
                            foldGutter: false,
                            indentUnit: 4,
                            tabSize: 4
                        }}
                    />
                    {error && <div className="error-message">{error}</div>}
                </div>
                    <div className="controls-container">
                        <div id="separatrix" />
                        <ThemeToggle />                        
                        <RunButton
                            isLoading={isLoading}
                            isRunning={isRunning}
                            onClick={handleRunClick}
                        />
                        <ReinitializeButton 
                            onClick={handleReinitialize}
                            disabled={isLoading}
                        />
                        <FPSInput 
                            onChange={setFPS}
                        />
                        <AutomatonSelect 
                            value={automatonType}
                            onChange={setAutomatonType}
                        />
                    </div>
            </div>

        </div>
    );
}

export default PyGrid;
