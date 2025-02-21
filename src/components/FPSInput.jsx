import React from "react";

function FPSInput({ onChange }) {
    const handleChange = (e) => {
        onChange(parseInt(e.target.value));
    };

    return (
        <input
            type="number"
            min="1"
            max="60"
            defaultValue="10"
            onBlur={handleChange}
            onKeyDown={(e) => {
                if (e.key === 'Enter') {
                    e.target.blur();
                }
            }}
            className="fps-input"
        />
    );
}

export default FPSInput;