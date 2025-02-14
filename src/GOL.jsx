import './style.css'
import { useEffect, useRef } from 'react';
import p5 from 'p5';


function GameOfLife(p) {
    let frame_rate = 10;
    let canvas_width = 700;
    let canvas_height = 700;
    let columnCount;
    let rowCount;
    let cellSize = 10;
    let currentCells = [];
    let nextCells = [];

    p.setup = function() {

        p.frameRate(frame_rate)
        p.createCanvas(canvas_width, canvas_height);

        // Calculate columns and rows
        columnCount = p.floor(canvas_width / cellSize);
        rowCount = p.floor(canvas_height / cellSize);

        for (let column = 0; column < columnCount; column++) {
            currentCells[column] = [];
        }

        for (let column = 0; column < columnCount; column++) {
        nextCells[column] = [];
        }

        p.noLoop()
    }

    p.draw = function() {
        generate();
        for (let column = 0; column < columnCount; column++) {
          for (let row = 0; row < rowCount; row++) {
            // Get cell value (0 or 1)
            let cell = currentCells[column][row];
            // Convert cell value to get black (0) for alive or white (255 (white) for dead
            p.fill((1 - cell) * 255);
            p.noStroke()
            p.square(column * cellSize, row * cellSize, cellSize);
          }
        }
      }
    

    p.mousePressed = function() {
        console.log("mouse pressed")
        randomizeBoard();
        p.loop();
    }
    
    function randomizeBoard() {
    for (let column = 0; column < columnCount; column++) {
        for (let row = 0; row < rowCount; row++) {
        // Randomly select value of either 0 (dead) or 1 (alive)
        currentCells[column][row] = p.random([0, 1]);
        }
      }
    }

    // Create a new generation
    function generate() {
        // Loop through every spot in our 2D array and count living neighbors
        for (let column = 0; column < columnCount; column++) {
        for (let row = 0; row < rowCount; row++) {
            // Column left of current cell
            // if column is at left edge, use modulus to wrap to right edge
            let left = (column - 1 + columnCount) % columnCount;
    
            // Column right of current cell
            // if column is at right edge, use modulus to wrap to left edge
            let right = (column + 1) % columnCount;
    
            // Row above current cell
            // if row is at top edge, use modulus to wrap to bottom edge
            let above = (row - 1 + rowCount) % rowCount;
    
            // Row below current cell
            // if row is at bottom edge, use modulus to wrap to top edge
            let below = (row + 1) % rowCount;
    
            // Count living neighbors surrounding current cell
            let neighbours =
            currentCells[left][above] +
            currentCells[column][above] +
            currentCells[right][above] +
            currentCells[left][row] +
            currentCells[right][row] +
            currentCells[left][below] +
            currentCells[column][below] +
            currentCells[right][below];
    
            // Rules of Life
            // 1. Any live cell with fewer than two live neighbours dies
            // 2. Any live cell with more than three live neighbours dies
            if (neighbours < 2 || neighbours > 3) {
            nextCells[column][row] = 0;
            // 4. Any dead cell with exactly three live neighbours will come to life.
            } else if (neighbours === 3) {
            nextCells[column][row] = 1;
            // 3. Any live cell with two or three live neighbours lives, unchanged, to the next generation.
            } else nextCells[column][row] = currentCells[column][row];
          }
        }
    
        // Swap the current and next arrays for next generation
        let temp = currentCells;
        currentCells = nextCells;
        nextCells = temp;
    }
}

function CA2D() {
    const p5ContainerRef = useRef();

    useEffect(() => {
        const p5Instance = new p5(GameOfLife, p5ContainerRef.current);

        return () => {
            p5Instance.remove();
        }
    }, []);

    return (
        <div 
            className="GameOfLife" 
            ref={p5ContainerRef}
        />
    );
}

export default CA2D;