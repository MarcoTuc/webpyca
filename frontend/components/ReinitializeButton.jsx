import React from "react";

function ReinitializeButton({ onClick, disabled }) {
    return (
        <button 
            className="reinitialize-button"
            onClick={onClick}
            disabled={disabled}
        >
            <span className="icon">&#8635;</span> {/* Circular icon with arrows */}
        </button>
    );
}

export default ReinitializeButton;