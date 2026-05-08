import { startTransition, useDeferredValue, useEffect, useMemo, useState } from 'react'
import { CircleMarker, MapContainer, Popup, TileLayer, useMapEvents } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

const EMPTY_FORM = {
  datasetName: '',
  datasetTitle: '',
  outputName: '',
  sourceUrl: '',
  changeSummary: '',
}

function slugify(value) {
  return value.toLowerCase().trim().replace(/[^a-z0-9._-]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '') || 'dataset'
}

function summarizePeriods(periods, options) {
  if (!periods.length) return 'Stress periods: all'
  const labels = periods
    .slice()
    .sort((a, b) => a - b)
    .map((period) => `SP ${period + 1}`)
  if (options.length && periods.length === options.length) return `Stress periods: all (${options.length})`
  return `Stress periods: ${labels.join(', ')}`
}

function buildSuggestedFields(dataset, controls, state) {
  if (!dataset) return EMPTY_FORM
  const periodSummary = summarizePeriods(state.periods, controls.periodOptions || [])
  const baseName = slugify(state.datasetSuggestion && state.datasetSuggestion !== '__new__' ? state.datasetSuggestion : dataset.name)
  const selectedTag =
    state.colorBy !== 'flux' && state.categoryValue
      ? `${state.colorBy}-${slugify(state.categoryValue)}`
      : `${state.selectedIds.length}-cells`
  const rateTag =
    state.rateMode === 'scale_percent'
      ? `${Math.abs(Number(state.newRate) || 0)}pct-${Number(state.newRate) < 0 ? 'reduction' : 'increase'}`
      : `set-${Number(state.newRate) || 0}`
  const ext = state.fluxSource === 'rch' ? '.rch' : '.wel'
  return {
    datasetName: baseName,
    datasetTitle: dataset.title,
    outputName: `${dataset.name}_${slugify(rateTag)}_${slugify(selectedTag)}${ext}`,
    sourceUrl: dataset.sourceUrl,
    changeSummary: `${periodSummary}; rate mode: ${state.rateMode}; new rate: ${state.newRate}; selected cells: ${state.selectedIds.length}`,
  }
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  const contentType = response.headers.get('content-type') || ''
  if (!contentType.includes('application/json')) {
    const text = await response.text()
    const snippet = text.slice(0, 160).replace(/\s+/g, ' ').trim()
    throw new Error(`Expected JSON from ${url}, got ${response.status} ${response.statusText}: ${snippet}`)
  }
  const payload = await response.json()
  if (!response.ok) {
    throw new Error(payload.error || payload.message || 'Request failed')
  }
  return payload
}

function IconLabel({ icon, children, tone = 'default' }) {
  return (
    <span className={`icon-label icon-label-${tone}`}>
      <i className={`icon icon-${icon}`} aria-hidden="true" />
      <span>{children}</span>
    </span>
  )
}

function categoryColor(name) {
  const palette = ['#0f766e', '#b45309', '#7c3aed', '#2563eb', '#be123c', '#4d7c0f']
  let hash = 0
  for (let i = 0; i < name.length; i += 1) hash = (hash * 31 + name.charCodeAt(i)) >>> 0
  return palette[hash % palette.length]
}

function fluxColor(value) {
  if (value < 0) return '#1d4ed8'
  if (value > 0) return '#b91c1c'
  return '#94a3b8'
}

function cellColor(cell, colorBy) {
  if (colorBy === 'GCD_Name') return categoryColor(cell.gcd)
  if (colorBy === 'PGMA_Name') return categoryColor(cell.pgma)
  return fluxColor(cell.flux)
}

function cellRadius(cell, selected) {
  const base = Math.min(14, Math.max(4, Math.round(Math.abs(cell.flux) / 25)))
  return selected ? base + 5 : base
}

function MapEventsBridge({ onZoomChange }) {
  useMapEvents({
    zoomend(event) {
      onZoomChange(event.target.getZoom())
    },
  })
  return null
}

function CollapsibleCard({ title, icon, tone = 'muted', collapsed, onToggle, children }) {
  return (
    <section className={`c-card control-card ${collapsed ? 'is-collapsed' : ''}`}>
      <div className="panel-head">
        <div className="panel-head__label">
          <h2 className="panel-title">{title}</h2>
          <IconLabel icon={icon} tone={tone}>{title}</IconLabel>
        </div>
        <button
          className="c-button c-button--tertiary c-button--size-small collapse-toggle"
          type="button"
          onClick={onToggle}
          aria-expanded={!collapsed}
          aria-label={collapsed ? `Expand ${title}` : `Minimize ${title}`}
        >
          <i className={`icon ${collapsed ? 'icon-expand' : 'icon-contract'}`} aria-hidden="true" />
        </button>
      </div>
      <div className={`collapsible-body ${collapsed ? 'is-collapsed' : ''}`}>
        <div className="collapsible-body__inner">{children}</div>
      </div>
    </section>
  )
}

export default function App() {
  const [datasets, setDatasets] = useState([])
  const [dataset, setDataset] = useState('')
  const [loaded, setLoaded] = useState(null)
  const [mapData, setMapData] = useState({ center: { lat: 30.26, lon: -97.74, zoom: 8 }, cells: [] })
  const [controls, setControls] = useState({ periodOptions: [], layerOptions: [], categoryOptions: [] })
  const [summary, setSummary] = useState({ selectedCount: 0, activeCellCount: 0, fluxSource: 'wel' })
  const [jwtToken, setJwtToken] = useState('')
  const [tapisUsername, setTapisUsername] = useState('')
  const [loginForm, setLoginForm] = useState({ username: '', password: '' })
  const [loginStatus, setLoginStatus] = useState('')
  const [statusMessage, setStatusMessage] = useState('')
  const [suggestions, setSuggestions] = useState([])
  const [selectionMode, setSelectionMode] = useState('manual')
  const [zoom, setZoom] = useState(8)
  const [isLoading, setIsLoading] = useState(false)
  const [formTouched, setFormTouched] = useState(false)
  const [workflowRun, setWorkflowRun] = useState(null)
  const [workflowGroupId, setWorkflowGroupId] = useState('')
  const [workflowRegistered, setWorkflowRegistered] = useState(false)
  const [showWorkflowModal, setShowWorkflowModal] = useState(false)
  const [workflowStatus, setWorkflowStatus] = useState('')
  const [collapsedCards, setCollapsedCards] = useState({
    dataset: true,
    mapControls: true,
    editSelection: true,
    publish: true,
  })
  const [viewState, setViewState] = useState({
    fluxSource: 'wel',
    colorBy: 'flux',
    colorPeriod: '',
    periods: [],
    layers: [1],
    rateMode: 'set',
    newRate: -20,
    addMissing: false,
    categoryValue: '',
    selectedIds: [],
    datasetSuggestion: '',
  })
  const [publishForm, setPublishForm] = useState(EMPTY_FORM)

  const deferredSelectedIds = useDeferredValue(viewState.selectedIds)

  useEffect(() => {
    requestJson('/api/datasets')
      .then((payload) => {
        setDatasets(payload.datasets || [])
        if (payload.datasets?.length) setDataset(payload.datasets[0].value)
      })
      .catch((error) => setStatusMessage(error.message))
  }, [])

  useEffect(() => {
    if (!tapisUsername || !jwtToken) {
      setSuggestions([{ label: 'New dataset', value: '__new__' }])
      return
    }
    requestJson(`/api/dataset-suggestions?username=${encodeURIComponent(tapisUsername)}&jwtToken=${encodeURIComponent(jwtToken)}`)
      .then((payload) => setSuggestions(payload.options || []))
      .catch(() => setSuggestions([{ label: 'New dataset', value: '__new__' }]))
  }, [jwtToken, tapisUsername])

  useEffect(() => {
    if (!loaded || formTouched) return
    setPublishForm(buildSuggestedFields(loaded, controls, viewState))
  }, [loaded, controls, viewState, formTouched])

  useEffect(() => {
    if (!workflowRun || !jwtToken) return undefined
    const interval = window.setInterval(() => {
      requestJson(
        `/api/workflow-runs/${encodeURIComponent(workflowRun.groupId)}/${encodeURIComponent(workflowRun.pipelineId)}/${encodeURIComponent(workflowRun.runId)}?jwtToken=${encodeURIComponent(jwtToken)}`,
      )
        .then((payload) => {
          const normalizedStatus = String(payload.status || '').toUpperCase()
          if (['COMPLETED', 'COMPLETE', 'SUCCESS', 'SUCCEEDED'].includes(normalizedStatus)) {
            const resultMessage = payload.result?.message || `Workflow run ${workflowRun.runId} completed.`
            setStatusMessage(resultMessage)
            setWorkflowRun(null)
          } else if (['FAILED', 'ERROR', 'CANCELED', 'CANCELLED'].includes(normalizedStatus)) {
            setStatusMessage(payload.result?.message || `Workflow run ${workflowRun.runId}: ${payload.status}`)
            setWorkflowRun(null)
          } else {
            setStatusMessage(`Workflow run ${workflowRun.runId}: ${payload.status}`)
          }
        })
        .catch((error) => {
          setStatusMessage(error.message)
          setWorkflowRun(null)
        })
    }, 4000)
    return () => window.clearInterval(interval)
  }, [workflowRun, jwtToken])

  useEffect(() => {
    if (!dataset) return
    setIsLoading(true)
    const params = new URLSearchParams({
      fluxSource: viewState.fluxSource,
      colorBy: viewState.colorBy,
    })
    if (viewState.colorPeriod !== '') params.set('colorPeriod', String(viewState.colorPeriod))
    if (viewState.periods.length) params.set('periods', viewState.periods.join(','))
    requestJson(`/api/datasets/${encodeURIComponent(dataset)}/view?${params.toString()}`)
      .then((payload) => {
        setLoaded(payload.dataset)
        setControls(payload.controls)
        setSummary({ ...payload.summary, selectedCount: deferredSelectedIds.length })
        setMapData(payload.mapData)
        setStatusMessage('')
        startTransition(() => {
          setViewState((current) => {
            const nextPeriods = current.periods.length ? current.periods : payload.controls.periodOptions.slice(0, 1).map((option) => option.value)
            const nextLayers = current.layers.length ? current.layers : payload.controls.layerOptions.slice(0, 1).map((option) => option.value)
            const nextColorPeriod =
              current.colorPeriod !== '' ? current.colorPeriod : payload.controls.colorPeriodOptions?.[0]?.value ?? ''
            return { ...current, periods: nextPeriods, layers: nextLayers, colorPeriod: nextColorPeriod }
          })
          if (payload.mapData?.center?.zoom) setZoom(payload.mapData.center.zoom)
        })
      })
      .catch((error) => setStatusMessage(error.message))
      .finally(() => setIsLoading(false))
  }, [dataset, viewState.fluxSource, viewState.colorBy, viewState.colorPeriod, viewState.periods, deferredSelectedIds.length])

  function updateForm(key, value) {
    setFormTouched(true)
    setPublishForm((current) => ({ ...current, [key]: value }))
  }

  async function loadCategorySelection() {
    if (!dataset || !viewState.categoryValue || viewState.colorBy === 'flux') return
    const payload = await requestJson(`/api/datasets/${encodeURIComponent(dataset)}/category-selection`, {
      method: 'POST',
      body: JSON.stringify({ colorBy: viewState.colorBy, categoryValue: viewState.categoryValue }),
    })
    setSelectionMode('category')
    setViewState((current) => ({ ...current, selectedIds: payload.selectedIds || [] }))
  }

  async function handleLogin(event) {
    event.preventDefault()
    setLoginStatus('Authenticating...')
    try {
      const payload = await requestJson('/api/login', {
        method: 'POST',
        body: JSON.stringify(loginForm),
      })
      setJwtToken(payload.jwtToken)
      setTapisUsername(payload.username)
      setLoginStatus(`Logged in as ${payload.username}`)
      setWorkflowRegistered(false)
      setWorkflowStatus('')
      setShowWorkflowModal(true)
    } catch (error) {
      setLoginStatus(error.message)
    }
  }

  async function handleWorkflowRegister(event) {
    event?.preventDefault?.()
    setWorkflowStatus('Registering workflow...')
    try {
      const payload = await requestJson('/api/workflow/register', {
        method: 'POST',
        body: JSON.stringify({
          jwtToken,
          workflowGroupId,
        }),
      })
      setWorkflowGroupId(payload.groupId)
      setWorkflowRegistered(true)
      setWorkflowStatus(payload.message)
      setShowWorkflowModal(false)
      setStatusMessage(payload.message)
    } catch (error) {
      setWorkflowRegistered(false)
      setWorkflowStatus(error.message)
    }
  }

  async function handleApply() {
    try {
      const payload = await requestJson('/api/apply', {
        method: 'POST',
        body: JSON.stringify({
          dataset,
          selectedIds: viewState.selectedIds,
          fluxSource: viewState.fluxSource,
          newRate: Number(viewState.newRate),
          rateMode: viewState.rateMode,
          addMissing: viewState.addMissing,
          layers: viewState.layers,
          periods: viewState.periods,
          jwtToken,
          tapisUsername,
          workflowGroupId,
          ...publishForm,
        }),
      })
      if (payload.mode === 'workflow') {
        setWorkflowRun({ groupId: payload.groupId, pipelineId: payload.pipelineId, runId: payload.runId })
      }
      setStatusMessage(payload.message)
    } catch (error) {
      setStatusMessage(error.message)
    }
  }

  function toggleSelected(cellId) {
    setSelectionMode('manual')
    setViewState((current) => {
      const exists = current.selectedIds.includes(cellId)
      return {
        ...current,
        selectedIds: exists ? current.selectedIds.filter((value) => value !== cellId) : [...current.selectedIds, cellId],
      }
    })
  }

  const selectedSet = useMemo(() => new Set(viewState.selectedIds), [viewState.selectedIds])
  const periodSummary = useMemo(() => summarizePeriods(viewState.periods, controls.periodOptions || []), [viewState.periods, controls.periodOptions])
  function toggleCard(key) {
    setCollapsedCards((current) => ({ ...current, [key]: !current[key] }))
  }

  return (
    <div className="c-page app-shell-react">
      <header className="app-header">
        <div className="app-header__intro">
          <p className="app-kicker">
            <IconLabel icon="applications">FloPy Interactive</IconLabel>
          </p>
          <h1 className="app-title">WEL/RCH Workbench</h1>
          <p className="app-subtitle">
            Leaflet handles mapping in the browser, and the backend can pass ETL-oriented apply and publish work through to Tapis Workflows.
          </p>
          <div className="app-header__meta">
            <span className="c-pill"><IconLabel icon="globe">Leaflet Map</IconLabel></span>
            <span className="c-pill"><IconLabel icon="project">Workflow Gateway</IconLabel></span>
            {workflowRegistered && workflowGroupId ? <span className="c-pill"><IconLabel icon="approved">Group {workflowGroupId}</IconLabel></span> : null}
          </div>
        </div>
        {jwtToken ? (
          <section className="c-card login-panel login-panel--compact">
            <div className="panel-head">
              <h2 className="panel-title">Logged In</h2>
              <IconLabel icon="approved" tone="live">Authenticated</IconLabel>
            </div>
            <p className="login-summary">Welcome, {tapisUsername}{workflowRegistered && workflowGroupId ? ` · ${workflowGroupId}` : ''}</p>
          </section>
        ) : (
          <form className="c-card login-panel" onSubmit={handleLogin}>
            <div className="panel-head">
              <h2 className="panel-title">Tapis Login</h2>
              <IconLabel icon="lock" tone="muted">Secure</IconLabel>
            </div>
            <label className="field-label" htmlFor="login-username">Username</label>
            <input id="login-username" value={loginForm.username} onChange={(event) => setLoginForm((current) => ({ ...current, username: event.target.value }))} placeholder="Tapis username" />
            <label className="field-label" htmlFor="login-password">Password</label>
            <input id="login-password" type="password" value={loginForm.password} onChange={(event) => setLoginForm((current) => ({ ...current, password: event.target.value }))} placeholder="Tapis password" />
            <button className="c-button c-button--primary c-button--width-long" type="submit">
              <i className="icon icon-unlock" aria-hidden="true" />
              <span>Login</span>
            </button>
            <div className="c-message c-message--compact login-message">
              <div className="c-message__body">{loginStatus || 'Workflow dispatch is enabled when a Tapis token is present.'}</div>
            </div>
          </form>
        )}
      </header>

      <main className="workspace-grid">
        <aside className="control-column">
          <CollapsibleCard
            title="Dataset"
            icon="data-files"
            collapsed={collapsedCards.dataset}
            onToggle={() => toggleCard('dataset')}
          >
            <label>Dataset</label>
            <select value={dataset} onChange={(event) => setDataset(event.target.value)}>
              {datasets.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            <div className="inline-stats stats-pills">
              <span className="c-pill">{summary.activeCellCount} active cells</span>
              <span className="c-pill">{viewState.selectedIds.length} selected</span>
            </div>
          </CollapsibleCard>

          <CollapsibleCard
            title="Map Controls"
            icon="bar-graph"
            collapsed={collapsedCards.mapControls}
            onToggle={() => toggleCard('mapControls')}
          >
            <label>Flux source</label>
            <select value={viewState.fluxSource} onChange={(event) => setViewState((current) => ({ ...current, fluxSource: event.target.value }))}>
              <option value="wel">Well</option>
              <option value="rch" disabled={!loaded?.hasRch}>Recharge</option>
            </select>
            <label>Color by</label>
            <select value={viewState.colorBy} onChange={(event) => setViewState((current) => ({ ...current, colorBy: event.target.value, categoryValue: '' }))}>
              {(controls.colorOptions || []).map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            {viewState.colorBy === 'flux' && (
              <>
                <label>Color period</label>
                <select value={viewState.colorPeriod} onChange={(event) => setViewState((current) => ({ ...current, colorPeriod: Number(event.target.value) }))}>
                  {(controls.colorPeriodOptions || []).map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </>
            )}
            {viewState.colorBy !== 'flux' && (
              <>
                <label>Category</label>
                <select value={viewState.categoryValue} onChange={(event) => setViewState((current) => ({ ...current, categoryValue: event.target.value }))}>
                  <option value="">Select category</option>
                  {(controls.categoryOptions || []).map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
                <button className="c-button c-button--secondary" type="button" onClick={loadCategorySelection}>
                  <i className="icon icon-search-folder" aria-hidden="true" />
                  <span>Select category</span>
                </button>
              </>
            )}
            <label>Stress periods</label>
            <select multiple value={viewState.periods.map(String)} onChange={(event) => setViewState((current) => ({ ...current, periods: [...event.target.selectedOptions].map((option) => Number(option.value)) }))}>
              {(controls.periodOptions || []).map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            <div className="c-message c-message--compact helper-message">
              <div className="c-message__body">{periodSummary}</div>
            </div>
          </CollapsibleCard>

          <CollapsibleCard
            title="Edit Selection"
            icon="edit-document"
            collapsed={collapsedCards.editSelection}
            onToggle={() => toggleCard('editSelection')}
          >
            <label>Rate mode</label>
            <select value={viewState.rateMode} onChange={(event) => setViewState((current) => ({ ...current, rateMode: event.target.value }))}>
              <option value="set">Set</option>
              <option value="scale_percent">Scale (%)</option>
            </select>
            <label>New rate</label>
            <input type="number" value={viewState.newRate} onChange={(event) => setViewState((current) => ({ ...current, newRate: Number(event.target.value) }))} />
            <label>Layers</label>
            <select multiple value={viewState.layers.map(String)} onChange={(event) => setViewState((current) => ({ ...current, layers: [...event.target.selectedOptions].map((option) => Number(option.value)) }))} disabled={viewState.fluxSource === 'rch'}>
              {(controls.layerOptions || []).map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            <label className="checkbox-row">
              <input type="checkbox" checked={viewState.addMissing} disabled={viewState.fluxSource === 'rch'} onChange={(event) => setViewState((current) => ({ ...current, addMissing: event.target.checked }))} />
              <span>Add missing wells</span>
            </label>
            <div className="selection-strip">
              <span className="c-pill">{selectionMode === 'category' ? 'Category selection' : 'Click markers to select'}</span>
              <button className="c-button c-button--tertiary c-button--width-short" type="button" onClick={() => setViewState((current) => ({ ...current, selectedIds: [] }))}>
                <i className="icon icon-close" aria-hidden="true" />
                <span>Clear</span>
              </button>
            </div>
          </CollapsibleCard>

          <CollapsibleCard
            title="Publish Update"
            icon="upload"
            collapsed={collapsedCards.publish}
            onToggle={() => toggleCard('publish')}
          >
            <label>Suggested dataset</label>
            <select value={viewState.datasetSuggestion} onChange={(event) => {
              const value = event.target.value
              setViewState((current) => ({ ...current, datasetSuggestion: value }))
              if (value && value !== '__new__') updateForm('datasetName', value)
            }}>
              {suggestions.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
            <label>Dataset name</label>
            <input value={publishForm.datasetName} onChange={(event) => updateForm('datasetName', event.target.value)} />
            <label>Dataset title</label>
            <input value={publishForm.datasetTitle} onChange={(event) => updateForm('datasetTitle', event.target.value)} />
            <label>Output filename</label>
            <input value={publishForm.outputName} onChange={(event) => updateForm('outputName', event.target.value)} />
            <label>Source URL</label>
            <input value={publishForm.sourceUrl} onChange={(event) => updateForm('sourceUrl', event.target.value)} />
            <label>Change summary</label>
            <textarea value={publishForm.changeSummary} onChange={(event) => updateForm('changeSummary', event.target.value)} rows={4} />
            <button className="c-button c-button--primary c-button--width-long" type="button" disabled={!jwtToken} onClick={handleApply}>
              <i className="icon icon-save" aria-hidden="true" />
              <span>Apply and publish</span>
            </button>
          </CollapsibleCard>
        </aside>

        <section className="map-column">
          <div className="c-card map-stage">
            <div className="stage-head">
              <div>
                <p className="app-kicker">
                  <IconLabel icon="globe" tone="muted">{loaded?.title || 'Dataset map'}</IconLabel>
                </p>
                <h2 className="panel-title map-title">{loaded?.name || 'Loading'}</h2>
              </div>
              <div className="stage-badges">
                <span className="c-pill">{isLoading ? 'Refreshing map' : 'Map ready'}</span>
                <span className="c-pill">{workflowRun ? `Workflow ${workflowRun.runId}` : 'Backend pass-through'}</span>
              </div>
            </div>
            <div className="map-frame leaflet-frame">
              <MapContainer center={[mapData.center.lat, mapData.center.lon]} zoom={zoom} className="leaflet-map" scrollWheelZoom>
                <MapEventsBridge onZoomChange={setZoom} />
                <TileLayer
                  attribution='&copy; OpenStreetMap contributors'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                {mapData.cells.map((cell) => {
                  const selected = selectedSet.has(cell.cellId)
                  return (
                    <CircleMarker
                      key={cell.cellId}
                      center={[cell.lat, cell.lon]}
                      radius={cellRadius(cell, selected)}
                      pathOptions={{
                        color: selected ? '#111827' : cellColor(cell, viewState.colorBy),
                        fillColor: cellColor(cell, viewState.colorBy),
                        fillOpacity: selected ? 0.95 : 0.7,
                        weight: selected ? 3 : 1,
                      }}
                      eventHandlers={{ click: () => toggleSelected(cell.cellId) }}
                    >
                      <Popup>
                        <strong>CELL_ID {cell.cellId}</strong><br />
                        ROW {cell.row} COL {cell.col}<br />
                        Flux {cell.flux}<br />
                        GCD {cell.gcd}<br />
                        PGMA {cell.pgma}
                      </Popup>
                    </CircleMarker>
                  )
                })}
              </MapContainer>
            </div>
            <div className="c-message status-ribbon">
              <div className="c-message__body">{statusMessage || 'Use category selection or click map markers, then submit an update directly or through a configured Tapis Workflow.'}</div>
            </div>
          </div>
        </section>
      </main>
      {showWorkflowModal ? (
        <div className="workflow-modal-backdrop" role="presentation">
          <div className="c-card workflow-modal" role="dialog" aria-modal="true" aria-labelledby="workflow-modal-title">
            <div className="panel-head">
              <div className="panel-head__label">
                <h2 className="panel-title" id="workflow-modal-title">Register Workflow</h2>
                <IconLabel icon="project" tone="live">First login setup</IconLabel>
              </div>
            </div>
            <p className="workflow-modal__body">Enter the workflow group you want this app to use. The backend will create the pipeline there if it does not exist yet.</p>
            <form onSubmit={handleWorkflowRegister}>
              <label className="field-label" htmlFor="workflow-group-id">Workflow group id</label>
              <input
                id="workflow-group-id"
                value={workflowGroupId}
                onChange={(event) => setWorkflowGroupId(event.target.value)}
                placeholder="your-workflow-group"
              />
              <div className="workflow-modal__actions">
                <button className="c-button c-button--primary" type="submit" disabled={!workflowGroupId.trim()}>
                  <i className="icon icon-save" aria-hidden="true" />
                  <span>Register Workflow</span>
                </button>
              </div>
            </form>
            <div className="c-message c-message--compact login-message">
              <div className="c-message__body">{workflowStatus || 'This only needs to be done once per workflow group.'}</div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
