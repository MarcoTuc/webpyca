import React from 'react';
import { Select, MenuItem, FormControl } from '@mui/material';
import { styled } from '@mui/material/styles';
import Tooltip from './Tooltip';
import { useTheme } from '../components/ThemeProvider';

// Styled components
const StyledFormControl = styled(FormControl)(({ theme }) => ({
  position: 'absolute',
  top: '93vh',
  transform: 'translate(-5vw)',
  minWidth: 80,
  width: 'fit-content',
}));

const StyledSelect = styled(Select)(({ theme }) => ({
  fontFamily: '"Space Mono", monospace',
  fontSize: '12px',
  backgroundColor: 'var(--bg-color)',
  color: 'var(--text-color)',
  border: '1px solid var(--text-color)',
  borderRadius: 0,
  '& .MuiOutlinedInput-notchedOutline': {
    border: 'none',
  },
  '&:hover': {
    backgroundColor: 'var(--hover-color)',
  },
  '& .MuiSelect-select': {
    padding: '4px 14px 4px 14px',
    textAlign: 'center',
  },
  '& .MuiInputBase-input': {
    textAlign: 'center',
  },
  '& .MuiOutlinedInput-input': {
    paddingLeft: '14px !important',
    paddingRight: '14px !important',
    display: 'flex',
    justifyContent: 'center',
  },
  '& .MuiSelect-icon': {
    right: '0px',
    display: 'none',
  }
}));

const StyledMenuItem = styled(MenuItem)(({ theme }) => ({
  fontFamily: '"Space Mono", monospace',
  fontSize: '12px',
  backgroundColor: 'var(--bg-color)',
  color: 'var(--text-color)',
  textAlign: 'center',
  justifyContent: 'center',
  '&:hover': {
    backgroundColor: 'var(--hover-color)',
  },
  '&.Mui-selected': {
    backgroundColor: 'var(--selection-color)',
  }
}));

function AutomatonSelect({ value, onChange }) {
  const { theme } = useTheme();
  
  return (
    <Tooltip content="Choose automaton">
      <StyledFormControl size="small">
        <StyledSelect
          value={value}
          onChange={(e) => onChange(e.target.value)}
          displayEmpty
          MenuProps={{
            PaperProps: {
              style: {
                backgroundColor: 'var(--bg-color)',
                borderRadius: 0,
                border: '1px solid var(--text-color)',
                textAlign: 'center'
              },
            },
            anchorOrigin: {
              vertical: 'top',
              horizontal: 'center',
            },
            transformOrigin: {
              vertical: 'bottom',
              horizontal: 'center',
            },
          }}
        >
          <StyledMenuItem value="lenia">Lenia</StyledMenuItem>
          <StyledMenuItem value="multilenia">RGB Lenia</StyledMenuItem>
          <StyledMenuItem value="gol">Game of Life</StyledMenuItem>
          <StyledMenuItem value="baralenia">Bara lenia</StyledMenuItem>
        </StyledSelect>
      </StyledFormControl>
    </Tooltip>
  );
}

export default AutomatonSelect;