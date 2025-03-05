import React from 'react';
import Tippy from '@tippyjs/react';
import 'tippy.js/dist/tippy.css';
import { useTheme } from './ThemeProvider';

function Tooltip({ children, content, ...props }) {
  const { theme } = useTheme();
  
  return (
    <Tippy 
      content={content}
      theme={theme}
      arrow={false}
      placement="right"
      offset={[0, 5]}
      animation="fade"
      duration={150}
      className={`custom-tooltip ${theme}`}
      {...props}
    >
      {children}
    </Tippy>
  );
}

export default Tooltip;