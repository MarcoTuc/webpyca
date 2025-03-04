import { useEffect, useState, useCallback } from "react";
import CodeMirror from "@uiw/react-codemirror"
import { python } from '@codemirror/lang-python'

import P5Wrapper from "./components/P5Wrapper";
import { useTheme } from './components/ThemeProvider';
import ThemeToggle from "./components/ThemeToggle";
import RunButton from "./components/RunButton";
import FPSInput from "./components/FPSInput";
import saved_automata from "./SavedAutomata";
import AutomatonSelect from "./components/AutomatonSelect";

// API base URL
const API_BASE_URL = "http://localhost:8000"; 

const initialCode = saved_automata.lenia;

function sketch(p, config, theme, isRGB) {
    
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
    
    // Add a variable to store the API communication function
    let sprayCallback = null;

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

    p.setSprayCallback = function(callback) {
        sprayCallback = callback;
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
            p.mouseY >= 0 && p.mouseY < grid.height && 
            sprayCallback) {
            // Convert mouse coordinates to grid coordinates accounting for transformation
            const mouseX = Math.floor((p.mouseX - grid.x) / (grid.cellSize * grid.scale));
            const mouseY = Math.floor((p.mouseY - grid.y) / (grid.cellSize * grid.scale));
            
            // Call the spray API through the callback
            sprayCallback(mouseY, mouseX);
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

    // New method to handle base64 encoded image data
    p.updateFromBase64 = function(base64Data) {
        if (!base64Data) return;
        
        const img = new Image();
        img.onload = function() {
            const canvas = document.createElement('canvas');
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageData.data;
            
            // Create 2D array from image data
            const cells = [];
            for (let y = 0; y < canvas.height; y++) {
                const row = [];
                for (let x = 0; x < canvas.width; x++) {
                    const idx = (y * canvas.width + x) * 4;
                    
                    if (isRGB) {
                        row.push([
                            data[idx] / 255,
                            data[idx + 1] / 255,
                            data[idx + 2] / 255
                        ]);
                    } else {
                        // For grayscale, just use the red channel
                        row.push(data[idx] / 255);
                    }
                }
                cells.push(row);
            }
            
            grid.cells = cells;
            
            // Update cell size if needed
            if (cells.length > 0) {
                const newCellSize = config.width / cells.length;
                if (newCellSize !== grid.cellSize) {
                    grid.cellSize = newCellSize;
                }
            }
            
            p.redraw();
        };
        
        img.src = `data:image/png;base64,${base64Data}`;
    }

    return p;
}

function TorchGrid({ themes }) {
    const { theme } = useTheme();
    const [p5Instance, setP5Instance] = useState(null);
    
    const [isLoading, setIsLoading] = useState(true);
    const [isRunning, setIsRunning] = useState(false);
    
    const [code, setCode] = useState(initialCode);
    
    const [error, setError] = useState(null);
    const [automatonType, setAutomatonType] = useState('lenia');
    
    const [canvasConfig, setCanvasConfig] = useState({
        width: 600,
        height: 600,
        cellSize: 1
    });

    const [FPS, setFPS] = useState(30);
    const [isRGB, setIsRGB] = useState(false);
    const [eventSource, setEventSource] = useState(null);

    const initializeSketch = useCallback((p) => {
        const instance = sketch(p, canvasConfig, theme, isRGB);
        
        // Set the spray callback
        instance.setSprayCallback((y, x) => {
            fetch(`${API_BASE_URL}/spray?x=${x}&y=${y}`, {
                method: 'POST'
            }).catch(err => {
                console.error("Failed to spray:", err);
            });
        });
        
        setP5Instance(instance);
        return instance;
    }, [canvasConfig, theme, isRGB]);

    useEffect(() => {
        if (p5Instance) {
            p5Instance.updateTheme(theme);
            p5Instance.redraw();
        }
    }, [theme, p5Instance]);

    useEffect(() => {
        if (p5Instance) {
            p5Instance.updateRGB(isRGB);
        }
    }, [isRGB, p5Instance]);

    function codeUpdater(newCode) {
        setCode(newCode);
    }

    // Initial setup - no need to load Pyodide anymore
    useEffect(() => {
        setIsLoading(false);
    }, []);

    // Handle automaton selection
    useEffect(() => {
        if (isRunning) {
            // Stop any existing stream
            stopStream();
        }
        
        // Fetch the actual implementation code instead of using saved_automata
        fetchImplementationCode(automatonType);
        selectAutomaton(automatonType);
    }, [automatonType]);

    // Cleanup event source on unmount
    useEffect(() => {
        return () => {
            stopStream();
        };
    }, []);

    const selectAutomaton = async (type) => {
        try {
            setError(null);
            setIsLoading(true);
            
            const response = await fetch(`${API_BASE_URL}/select`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    automata_type: type,
                    size: 256 // You can make this configurable
                })
            });
            
            if (!response.ok) {
                throw new Error(`Failed to select automaton: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.data) {
                // Update grid with initial state
                if (p5Instance) {
                    p5Instance.updateFromBase64(data.data);
                }
                
                // Check if the grid is RGB
                setIsRGB(type === 'multilenia');
            }
            
            setIsLoading(false);
        } catch (err) {
            setError(`Error selecting automaton: ${err.message}`);
            setIsLoading(false);
        }
    };

    const runStep = async () => {
        try {
            setError(null);
            
            const response = await fetch(`${API_BASE_URL}/step`);
            
            if (!response.ok) {
                throw new Error(`Failed to run step: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.data && p5Instance) {
                p5Instance.updateFromBase64(data.data);
            }
        } catch (err) {
            setError(`Error running step: ${err.message}`);
            setIsRunning(false);
        }
    };

    const startStream = () => {
        stopStream(); // Close any existing stream
        
        const source = new EventSource(`${API_BASE_URL}/run?fps=${FPS}`);
        
        // Handle different event types
        source.addEventListener('connected', (event) => {
            console.log("SSE connection established");
        });
        
        source.addEventListener('update', (event) => {
            if (p5Instance && event.data) {
                p5Instance.updateFromBase64(event.data);
            }
        });
        
        source.addEventListener('error', (event) => {
            setError(event.data);
            stopStream();
        });
        
        source.addEventListener('stats', (event) => {
            try {
                const stats = JSON.parse(event.data);
                // Optionally display the actual FPS somewhere in your UI
                console.log(`Actual FPS: ${stats.actual_fps}`);
            } catch (err) {
                console.error("Error parsing stats data:", err);
            }
        });
        
        // Also keep the general error handler
        source.onerror = (err) => {
            console.error("EventSource error:", err);
            setError("Connection to server lost");
            stopStream();
        };
        
        setEventSource(source);
    };

    const stopStream = () => {
        if (eventSource) {
            eventSource.close();
            setEventSource(null);
            
            // Also tell the server to stop
            fetch(`${API_BASE_URL}/stop`).catch(err => {
                console.error("Error stopping stream:", err);
            });
        }
    };

    const handleRunClick = () => {
        if (!isRunning) {
            startStream();
            setIsRunning(true);
        } else {
            stopStream();
            setIsRunning(false);
        }
    };

    // Handle FPS changes
    useEffect(() => {
        if (isRunning) {
            // Restart the stream with new FPS
            stopStream();
            startStream();
        }
    }, [FPS]);

    // Add this new function to fetch implementation code
    const fetchImplementationCode = async (type) => {
        try {
            const response = await fetch(`${API_BASE_URL}/get_implementation_code/${type}`);
            
            if (!response.ok) {
                throw new Error(`Failed to fetch implementation code: ${response.statusText}`);
                // Fall back to the saved code if the fetch fails
                setCode(saved_automata[type]);
                return;
            }
            
            const data = await response.json();
            if (data.code) {
                setCode(data.code);
            } else {
                // Fall back to the saved code if the response doesn't contain code
                setCode(saved_automata[type]);
            }
        } catch (err) {
            console.error(`Error fetching implementation code: ${err.message}`);
            // Fall back to the saved code if there's an error
            setCode(saved_automata[type]);
        }
    };

    // For binary mode
    async function fetchBinaryData() {
        const response = await fetch('/step');
        const arrayBuffer = await response.arrayBuffer();
        
        // Parse the header
        const headerView = new DataView(arrayBuffer);
        const ndims = headerView.getUint8(0);
        const dtype = headerView.getUint32(1, false);
        
        let offset = 5;
        const shape = [];
        for (let i = 0; i < ndims; i++) {
            shape.push(headerView.getUint32(offset, false));
            offset += 4;
        }
        
        // Extract the data
        const dataBuffer = arrayBuffer.slice(offset);
        
        // Create typed array based on dtype
        let dataArray;
        switch(dtype) {
            case 11: // float32
                dataArray = new Float32Array(dataBuffer);
                break;
            case 12: // float64
                dataArray = new Float64Array(dataBuffer);
                break;
            case 1: // int8
                dataArray = new Int8Array(dataBuffer);
                break;
            case 2: // uint8
                dataArray = new Uint8Array(dataBuffer);
                break;
            // Add other cases as needed
            default:
                dataArray = new Uint8Array(dataBuffer);
        }
        
        // Now you have the raw data in dataArray and its shape
        // You can render it directly to canvas or process as needed
    }

    // For streaming binary data
    function startBinaryStream() {
        const xhr = new XMLHttpRequest();
        xhr.open('GET', '/run?fps=30', true);
        xhr.responseType = 'arraybuffer';
        
        let buffer = new Uint8Array(0);
        
        xhr.onprogress = function() {
            const newData = new Uint8Array(xhr.response);
            if (newData.length > buffer.length) {
                // We have new data
                const newPortion = newData.slice(buffer.length);
                buffer = newData;
                
                // Process frames from newPortion
                processNewFrames(newPortion);
            }
        };
        
        xhr.send();
    }

    function processNewFrames(data) {
        let offset = 0;
        while (offset + 4 <= data.length) {
            // Read frame size
            const view = new DataView(data.buffer, offset);
            const frameSize = view.getUint32(0, false);
            offset += 4;
            
            // Check if we have the complete frame
            if (offset + frameSize <= data.length) {
                // Extract and process the frame
                const frameData = data.slice(offset, offset + frameSize);
                renderFrame(frameData);
                offset += frameSize;
            } else {
                break; // Wait for more data
            }
        }
    }

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

export default TorchGrid;
