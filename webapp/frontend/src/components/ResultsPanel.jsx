import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { CheckCircle, AlertCircle, Loader, Target } from 'lucide-react'

export default function ResultsPanel({ results, loading }) {
  const containerVariants = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.2,
      },
    },
  }

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    show: { opacity: 1, y: 0 },
  }

  if (loading && !results) {
    return (
      <div className="glass-effect p-6 h-full flex flex-col items-center justify-center">
        <div className="relative w-16 h-16 mb-4">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-full opacity-20 animate-pulse"></div>
          <Loader className="w-16 h-16 text-cyan-400 animate-spin absolute inset-0" />
        </div>
        <p className="text-slate-300 text-center">
          <span className="block font-semibold mb-1">Running ADGENT Analysis</span>
          <span className="text-sm text-slate-400">Agents are discussing...</span>
        </p>
      </div>
    )
  }

  if (!results) {
    return (
      <div className="glass-effect p-6 h-full flex flex-col items-center justify-center text-center">
        <Target className="w-12 h-12 text-slate-600 mb-3" />
        <p className="text-slate-400">Upload MRI and clinical data to begin analysis</p>
      </div>
    )
  }

  const { prediction, final_consensus } = results
  const isAD = prediction.diagnosis === 'AD'

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className="glass-effect p-6 h-full flex flex-col"
    >
      <h2 className="text-xl font-bold text-cyan-400 mb-6 flex items-center gap-2">
        <CheckCircle className="w-5 h-5" />
        Results
      </h2>

      {/* Main Diagnosis */}
      <motion.div
        variants={itemVariants}
        className={`p-4 rounded-lg mb-6 border-2 ${
          isAD
            ? 'bg-red-500/10 border-red-500/50'
            : 'bg-green-500/10 border-green-500/50'
        }`}
      >
        <p className="text-xs font-semibold text-slate-400 uppercase mb-1">
          Primary Diagnosis
        </p>
        <p
          className={`text-3xl font-bold ${
            isAD ? 'text-red-400' : 'text-green-400'
          }`}
        >
          {prediction.diagnosis}
        </p>
        <p className="text-sm text-slate-300 mt-2">
          Confidence: <span className="font-semibold">{prediction.confidence}</span>
        </p>
      </motion.div>

      {/* Probability Gauge */}
      <motion.div variants={itemVariants} className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-slate-300">p(AD) Score</span>
          <span className="text-lg font-bold text-cyan-400">
            {(prediction.p_ad * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full h-3 bg-slate-800 rounded-full overflow-hidden border border-white/10">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${prediction.p_ad * 100}%` }}
            transition={{ delay: 0.3, duration: 1 }}
            className={`h-full bg-gradient-to-r ${
              isAD ? 'from-red-500 to-red-600' : 'from-green-500 to-green-600'
            }`}
          ></motion.div>
        </div>
        <div className="flex justify-between text-xs text-slate-400 mt-1">
          <span>p(CN) {(prediction.p_cn * 100).toFixed(1)}%</span>
          <span>p(AD) {(prediction.p_ad * 100).toFixed(1)}%</span>
        </div>
      </motion.div>

      {/* Grad-CAM preview */}
      {results.agent_reasoning && results.agent_reasoning.gradcam && results.agent_reasoning.gradcam.gradcam_image && (
        <motion.div variants={itemVariants} className="mb-6">
          <p className="text-xs font-semibold text-slate-400 uppercase mb-2">Imaging Evidence (Grad-CAM)</p>
          <div className="w-full rounded-lg overflow-hidden border border-white/10">
            <img src={results.agent_reasoning.gradcam.gradcam_image} alt="Grad-CAM overview" className="w-full h-auto block" />
          </div>
        </motion.div>
      )}

      {/* Rationale */}
      <motion.div variants={itemVariants} className="bg-white/5 rounded-lg p-4 mb-6">
        <p className="text-xs font-semibold text-slate-400 uppercase mb-2">
          Clinical Rationale
        </p>
        <p className="text-sm text-slate-300 leading-relaxed">
          {final_consensus.rationale}
        </p>
      </motion.div>

      {/* Recommendation */}
      <motion.div
        variants={itemVariants}
        className="mt-auto p-4 rounded-lg bg-blue-500/10 border border-blue-500/30"
      >
        <p className="text-xs font-semibold text-blue-300 uppercase mb-2">
          Clinical Recommendation
        </p>
        <p className="text-sm text-blue-100 leading-relaxed">
          {final_consensus.recommendation}
        </p>
      </motion.div>
    </motion.div>
  )
}
