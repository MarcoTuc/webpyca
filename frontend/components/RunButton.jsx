import React from "react";

function RunButton( { isLoading, isRunning, onClick } ) {
    return (
        <button 
            className={`run-button ${isLoading ? 'loading' : ''}`}
            onClick={onClick}
            disabled={isLoading}
        >
            {isLoading ? 'Loading...' : isRunning ? 'Stop' : 'Run'}
        </button>
    )
}

export default RunButton;