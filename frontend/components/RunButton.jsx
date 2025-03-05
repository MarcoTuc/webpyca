import React from "react";
import Tooltip from "./Tooltip";

function RunButton( { isLoading, isRunning, onClick } ) {
    return (
        <Tooltip 
            content={isLoading ? "Loading Pyodide..." : isRunning ? "Stop simulation" : "Run simulation"}
        >
            <button 
                className={`run-button ${isLoading ? 'loading' : ''}`}
                onClick={onClick}
                disabled={isLoading}
            >
                {isLoading ? 'Loading...' : isRunning ? 'Stop' : 'Run'}
            </button>
        </Tooltip>
    )
}

export default RunButton;