import { useEffect } from 'react';
import './style.css'
import PyGrid from './PyGrid'
import { ThemeProvider } from './components/ThemeProvider'
import ThemeToggle from './components/ThemeToggle'
import { atomone } from '@uiw/codemirror-theme-atomone'
import { heartheme } from "./theme/CustomTheme";
import { preventZoom } from './utils/preventZoom';

function App() {
  // useEffect(() => {
  //   preventZoom();
  // }, []);

  return (
    <ThemeProvider>
      <div>
        <PyGrid themes={{ dark: atomone, light: heartheme }}/>
      </div>
    </ThemeProvider>
  )
}

export default App
