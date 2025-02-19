import P5Wrapper from "./P5Wrapper";

function GridMaker(p) {

    let canvas_width = 700
    let canvas_height = 700;
    let columnCount = array.length;
    let rowCount = array[0].length
    let cellSize = 10;
    let currentCells = [];
    let nextCells = [];

    p.setup = function() {
        p.createCanvas(canvas_width, canvas_height);
        for (let column = 0; column < columnCount; column++) {
            currentCells[column] = [];
        }
        for (let column = 0; column < columnCount; column++) {
        nextCells[column] = [];
        }
        p.noLoop()
    }

    p.draw = Grid.draw = function() {
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
}

function Grid({ array }) {
    return  <P5Wrapper 
                id = "grid-container"
                sketch={GridMaker(array)}
            /> 
}