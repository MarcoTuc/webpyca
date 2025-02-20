import './style.css'
import PyGrid from './PyGrid'
import { ThemeProvider } from './components/ThemeProvider'
import ThemeToggle from './components/ThemeToggle'

function App() {
  return (
    <ThemeProvider>
      <div>
        <ThemeToggle />
        <PyGrid />
      </div>
    </ThemeProvider>
  )
}

export default App
