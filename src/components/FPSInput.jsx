import React from "react";

function FPSInput({ onChange }) {
    return (
        <input
            type="number"
            min="1"
            max="120"
            defaultValue="10"
            onBlur={(e) => onChange(parseInt(e.target.value))}
            onKeyDown={(e) => {
                if (e.key === "Enter") {
                    e.target.blur();
                }
            }}
            className="fps-input"
        />
    );
}

export default FPSInput;