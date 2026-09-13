/**
 * Punto de entrada principal de la aplicación React (Vite).
 * 
 * Inyecta el componente principal <App /> en el div #root de index.html
 * y aplica los estilos globales definidos en index.css y App.css.
 */
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import './App.css';
import App from './App.jsx';

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);