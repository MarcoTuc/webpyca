import React from "react";
import Tooltip from "./Tooltip";

function ReinitializeButton({ onClick, disabled }) {
    return (
        <Tooltip content="Reset simulation to initial state">
            <button 
                className="reinitialize-button"
                onClick={onClick}
                disabled={disabled}
            >
                <span className="icon">&#8635;</span> {/* Circular icon with arrows */}
            </button>
        </Tooltip>
    );
}

export default ReinitializeButton;