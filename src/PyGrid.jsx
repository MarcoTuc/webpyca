import { useEffect, useState, useCallback } from "react";
import { loadPyodide } from "pyodide";

import CodeMirror from "@uiw/react-codemirror"
import { python } from '@codemirror/lang-python'

import P5Wrapper from "./components/P5Wrapper";
import { useTheme } from './components/ThemeProvider';
import ThemeToggle from "./components/ThemeToggle";

const initialImports =  
`
import numpy as np
import time

`
const initialCode = `



import numpy as np

class Automaton:
    def __init__(self, size):
        # Initialize with random binary state (0 or 1)
        self.grid = np.random.choice([0, 1], size=(size, size), p=[0.85, 0.15])
        self.radius = 12
        self.density =  0.6
    
    def draw(self):
        # Calculate the number of live neighbors for each cell using convolution
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        # Count neighbors using convolution with periodic boundaries
        neighbors = sum(
            np.roll(np.roll(self.grid, i, 0), j, 1) * kernel[i+1, j+1]
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
            if not (i == 0 and j == 0)
        )
        
        # Apply Conway's Game of Life rules:
        # 1. Live cell with 2 or 3 neighbors survives
        # 2. Dead cell with exactly 3 neighbors becomes alive
        # 3. All other cells die or stay dead
        self.grid = ((neighbors == 3) | (self.grid & (neighbors == 2))).astype(int)
        
        return self.grid
      
    def spray(self, x, y):
        # Define the square region bounds with periodic boundary handling
        size = self.grid.shape[0]
        r = self.radius
        
        # Create a grid of coordinates relative to center
        y_coords, x_coords = np.ogrid[-r:r+1, -r:r+1]
        # Calculate distances from center for each point
        distances = np.sqrt(x_coords**2 + y_coords**2)
        # Create circular mask with random density
        spray_pattern = (distances <= r) & (np.random.random((2*r + 1, 2*r + 1)) < self.density)
        
        # Apply spray pattern with periodic boundaries
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if spray_pattern[i+r, j+r]:
                    # Use modulo for periodic boundaries
                    new_x = (x + i) % size
                    new_y = (y + j) % size
                    self.grid[new_x, new_y] = 1

auto = Automaton(300)

def main():
    return auto.draw()

def spray(x, y):
    auto.spray(x, y)


`

function sketch(p, config, pyodideInstance) {
    
    let currentCells = [];
    let canvasWidth = config.width;
    let canvasHeight = config.height;
    let cellSize = config.cellSize;

    p.setup = function() {
        p.createCanvas(canvasWidth, canvasHeight, p.P2D);
        p.noLoop();
        p.pixelDensity(1);
        // p.noSmooth();
    }

    p.updateConfig = function(newConfig) {
        canvasWidth = newConfig.width;
        canvasHeight = newConfig.height;
        cellSize = newConfig.cellSize;
    }

    p.mousePressed = function() {
        handleMouse();
    }

    p.mouseDragged = function() {
        handleMouse();
    }

    function handleMouse() {
        // Only register mouse input if coordinates are within canvas
        if (p.mouseX >= 0 && p.mouseX < canvasWidth && 
            p.mouseY >= 0 && p.mouseY < canvasHeight) {
            const mouseX = Math.floor(p.mouseX / cellSize);
            const mouseY = Math.floor(p.mouseY / cellSize);
            
            // Call the Python handle_mouse function if it exists
            if (pyodideInstance) {
                
                try {
                    pyodideInstance.runPython(`
                        if 'spray' in globals() and callable(spray):
                            spray(${mouseY}, ${mouseX})
                    `);
                    console.log(mouseX, mouseY)
                } catch (err) {
                    console.error("Failed to call spray:", err);
                }
            }
        }
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

function PyGrid({ themes }) {
    const { theme } = useTheme();
    const [pyodideInstance, setPyodideInstance] = useState(null);
    const [p5Instance, setP5Instance] = useState(null);
    
    const [isLoading, setIsLoading] = useState(true);
    const [isRunning, setIsRunning] = useState(false);
    
    const [imports, setImports] = useState(initialImports);
    const [code, setCode] = useState(initialCode);
    
    // const [codeChanged, setCodeChanged] = useState(true);
    const [error, setError] = useState(null);

    const [canvasConfig, setCanvasConfig] = useState({
        width: 600,
        height: 600,
        cellSize: 1
    });

    const [storedCode, setStoredCode] = useState("");

    // useEffect(() => {
    //     console.log(p5Instance)
    // }, [p5Instance])

    const initializeSketch = useCallback((p) => {
        const instance = sketch(p, canvasConfig, pyodideInstance);
        setP5Instance(instance);
        return instance;
    }, [pyodideInstance, canvasConfig]);

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

    return (
        <div className="ascii-play-container">
            <div className="left-panel">
                <div className="editor-section">
                    <CodeMirror
                        value={code}
                        onChange={codeUpdater}
                        theme={themes[theme]}
                        extensions={[python()]}
                        basicSetup={{
                            lineNumbers: false,
                            foldGutter: false
                        }}
                    />
                    <ThemeToggle />
                    <div className="controls-container">
                        <button 
                            className={`run-button ${isLoading ? 'loading' : ''}`}
                            onClick={handleRunClick}
                            disabled={isLoading}
                        >
                            {isLoading ? 'Loading...' : isRunning ? 'Stop' : 'Run'}
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
