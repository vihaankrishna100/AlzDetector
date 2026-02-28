import React from 'react'
import { motion } from 'framer-motion'
import { Upload, FileUp, Zap } from 'lucide-react'

export default function InputPanel({
  niftiFile,
  clinicalData,
  loading,
  onNiftiUpload,
  onClinicalChange,
  onSubmit,
  onReset,
  fileInputRef,
}) {
  const handleDragOver = (e) => {
    e.preventDefault()
    e.currentTarget.classList.add('border-cyan-500', 'bg-cyan-500/10')
  }

  const handleDragLeave = (e) => {
    e.currentTarget.classList.remove('border-cyan-500', 'bg-cyan-500/10')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.currentTarget.classList.remove('border-cyan-500', 'bg-cyan-500/10')
    const files = e.dataTransfer.files
    if (files.length > 0) {
      onNiftiUpload(files[0])
    }
  }

  return (
    <div className="glass-effect p-6 h-full flex flex-col">
      <h2 className="text-xl font-bold text-cyan-400 mb-6 flex items-center gap-2">
        <Upload className="w-5 h-5" />
        Input Data
      </h2>

      {/* MRI Upload */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className="border-2 border-dashed border-slate-600 rounded-lg p-6 mb-6 text-center cursor-pointer transition-all hover:border-cyan-500 hover:bg-cyan-500/5"
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".nii.gz,.nii"
          onChange={(e) => e.target.files && onNiftiUpload(e.target.files[0])}
          className="hidden"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="w-full"
        >
          <FileUp className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
          <p className="text-sm font-medium text-slate-300">
            {niftiFile ? (
              <span className="text-cyan-400">{niftiFile.name}</span>
            ) : (
              <>Click or drag MRI file (.nii.gz)</>
            )}
          </p>
        </button>
      </div>

      {/* Clinical Data Form */}
      <div className="space-y-4 flex-1">
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Age (years)
          </label>
          <input
            type="number"
            value={clinicalData.age}
            onChange={(e) => onClinicalChange('age', e.target.value)}
            placeholder="e.g., 72"
            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
            disabled={loading}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            ADAS-Cog 13 Score
          </label>
          <input
            type="number"
            value={clinicalData.adas_cog_13}
            onChange={(e) => onClinicalChange('adas_cog_13', e.target.value)}
            placeholder="e.g., 18"
            step="0.01"
            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
            disabled={loading}
          />
          <p className="text-xs text-slate-400 mt-1">0-70 scale</p>
        </div>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            APOE4 Copies
          </label>
          <select
            value={clinicalData.apoe4_copies}
            onChange={(e) => onClinicalChange('apoe4_copies', e.target.value)}
            className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-white focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500"
            disabled={loading}
          >
            <option value="0">0 copies</option>
            <option value="1">1 copy</option>
            <option value="2">2 copies</option>
          </select>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 mt-6">
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onSubmit}
          disabled={loading || !niftiFile}
          className="flex-1 px-4 py-3 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-600 hover:to-blue-700 disabled:from-slate-700 disabled:to-slate-600 text-white font-semibold rounded-lg transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Zap className="w-4 h-4" />
          {loading ? 'Analyzing...' : 'Analyze'}
        </motion.button>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onReset}
          disabled={loading}
          className="flex-1 px-4 py-3 bg-slate-700 hover:bg-slate-600 text-white font-semibold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed"
        >
          Reset
        </motion.button>
      </div>
    </div>
  )
}
