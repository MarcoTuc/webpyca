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

const initialImports =  
`
import numpy as np
import time

`
const initialCode = saved_automata.lenia

function sketch(p, config, pyodideInstance, theme) {
    
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
                const value = grid.cells[i][j];
                // Interpolate between light and dark background colors
                const r = theme === 'dark' 
                    ? Math.floor(value * 255 + (1 - value) * darkBg.r)
                    : Math.floor(value * darkBg.r + (1 - value) * lightBg.r);
                const g = theme === 'dark'
                    ? Math.floor(value * 255 + (1 - value) * darkBg.g)
                    : Math.floor(value * darkBg.g + (1 - value) * lightBg.g);
                const b = theme === 'dark'
                    ? Math.floor(value * 255 + (1 - value) * darkBg.b)
                    : Math.floor(value * darkBg.b + (1 - value) * lightBg.b);
                
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

    const initializeSketch = useCallback((p) => {
        const instance = sketch(p, canvasConfig, pyodideInstance, theme);
        setP5Instance(instance);
        return instance;
    }, [pyodideInstance, canvasConfig]);

    useEffect(() => {
        if (p5Instance) {
            p5Instance.updateTheme(theme)
            p5Instance.redraw()
        }
    }, [theme])

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
                const micropip = await pyodide.pyimport("micropip");
                await micropip.install("numpy");
                await micropip.install("autograd")
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

    const runCode = async () => {

        if (!pyodideInstance || !p5Instance) return;
        
        try {
            setError(null);
            
            // Check if code has changed from stored version
            if (code !== storedCode) {
                // Run one step to get canvas size
                await pyodideInstance.runPythonAsync(code);
                
                const hasMainFunction = await pyodideInstance.runPythonAsync(`
                    'main' in globals() and callable(globals()['main'])
                `);
                
                if (!hasMainFunction) {
                    setError("No main() function defined");
                    return;
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
                    
                    // Store the new code and reinitialize
                    setStoredCode(code);
                    await pyodideInstance.runPythonAsync(code);
                } else {
                    setError("main() must return an array");
                    return;
                }
            }
            
            // Normal execution
            const result = await pyodideInstance.runPythonAsync("main()");
            const jsResult = result.toJs();
            
            if (Array.isArray(jsResult)) {       
                console.log("received array from Python")   
                p5Instance.updateCells(jsResult);
            } else {
                setError("main() must return an array");
            }
        } catch (err) {
            setError(err.message);
            setIsRunning(false);
        }
    };

    const handleRunClick = (e) => {
        if (!isRunning) {
            runCode().then(() => {
                if (!error) setIsRunning(true);
            });
        } else {
            setIsRunning(false);
        }
    };

    useEffect(() => {
        setCode(saved_automata[automatonType]);
        setIsRunning(false);
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
