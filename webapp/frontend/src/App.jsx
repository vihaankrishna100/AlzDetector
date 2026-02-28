import React, { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import toast, { Toaster } from 'react-hot-toast'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Brain,
  Upload,
  Activity,
  Shield,
  CheckCircle,
  AlertCircle,
  Zap,
  MessageCircle,
  FileUp,
} from 'lucide-react'
import InputPanel from './components/InputPanel'
import ResultsPanel from './components/ResultsPanel'
import AgentPanel from './components/AgentPanel'
import Header from './components/Header'

export default function App() {
  const [loading, setLoading] = useState(false)
  const [niftiFile, setNiftiFile] = useState(null)
  const [clinicalData, setClinicalData] = useState({
    age: '',
    adas_cog_13: '',
    apoe4_copies: '0',
  })
  const [results, setResults] = useState(null)
  const [agentReasoning, setAgentReasoning] = useState(null)
  const fileInputRef = useRef(null)

  const handleNiftiUpload = (file) => {
    if (file && file.name.endsWith('.nii.gz')) {
      setNiftiFile(file)
      toast.success('MRI file uploaded successfully')
    } else {
      toast.error('Please upload a .nii.gz file')
    }
  }

  const handleClinicalChange = (field, value) => {
    setClinicalData(prev => ({
      ...prev,
      [field]: value,
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!niftiFile) {
      toast.error('Please upload an MRI file')
      return
    }

    if (!clinicalData.age || !clinicalData.adas_cog_13) {
      toast.error('Please fill in all clinical data')
      return
    }

    setLoading(true)
    const toastId = toast.loading('Running ADGENT inference...')

    try {
      const formData = new FormData()
      formData.append('nifti_file', niftiFile)
      formData.append('age', parseFloat(clinicalData.age))
      formData.append('adas_cog_13', parseFloat(clinicalData.adas_cog_13))
      formData.append('apoe4_copies', parseInt(clinicalData.apoe4_copies))

      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      setResults(response.data)
      setAgentReasoning(response.data.agent_reasoning)
      toast.success('Inference complete!', { id: toastId })
    } catch (error) {
      console.error(error)
      toast.error(
        error.response?.data?.detail || 'Inference failed. Make sure backend is running.',
        { id: toastId }
      )
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    setNiftiFile(null)
    setResults(null)
    setAgentReasoning(null)
    setClinicalData({
      age: '',
      adas_cog_13: '',
      apoe4_copies: '0',
    })
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-blue-950 to-slate-950">
      <Toaster position="top-right" />
      <Header />

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column: Input */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-1"
          >
            <InputPanel
              niftiFile={niftiFile}
              clinicalData={clinicalData}
              loading={loading}
              onNiftiUpload={handleNiftiUpload}
              onClinicalChange={handleClinicalChange}
              onSubmit={handleSubmit}
              onReset={handleReset}
              fileInputRef={fileInputRef}
            />
          </motion.div>

          {/* Middle Column: Results */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-1"
          >
            <ResultsPanel results={results} loading={loading} />
          </motion.div>

          {/* Right Column: Agent Reasoning */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="lg:col-span-1"
          >
            <AgentPanel reasoning={agentReasoning} loading={loading} />
          </motion.div>
        </div>
      </div>
    </div>
  )
}
