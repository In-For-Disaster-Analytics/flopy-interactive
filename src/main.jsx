import React from 'react'
import ReactDOM from 'react-dom/client'

import '@tacc/core-styles/dist/core-styles.base.css'
import '@tacc/core-styles/dist/core-styles.portal.css'
import App from './App'
import './app.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
