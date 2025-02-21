import React from 'react';

function AutomatonSelect({ value, onChange }) {
    return (
        <select 
            className="automaton-select"
            value={value}
            onChange={(e) => onChange(e.target.value)}
        >
            <option value="lenia">Lenia</option>
            <option value="gol">Game of Life</option>
        </select>
    );
}

export default AutomatonSelect;