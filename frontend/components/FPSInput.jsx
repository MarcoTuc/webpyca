import React from "react";
import Tooltip from "./Tooltip";

function FPSInput({ onChange }) {
    const handleChange = (e) => {
        onChange(parseInt(e.target.value));
    };

    return (
        <Tooltip content="Set frames per second (1-60)">
            <input
                type="number"
                min="1"
                max="60"
                defaultValue="30"
                onBlur={handleChange}
                onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                        e.target.blur();
                    }
                }}
                className="fps-input"
            />
        </Tooltip>
    );
}

export default FPSInput;