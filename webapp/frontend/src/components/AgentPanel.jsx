import React, { useState } from 'react'

import { motion, AnimatePresence } from 'framer-motion'

import { MessageCircle, ChevronDown, Brain, Heart, Microscope, CheckCircle } from 'lucide-react'

const AgentCard = ({ agentData, icon: Icon, color }) => {
  const [expanded, setExpanded] = useState(false)
  const agent = agentData.agent
  const analysis = agentData.analysis

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="mb-3 glass-effect overflow-hidden"

    >
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-start gap-3 hover:bg-white/5 transition-colors text-left"
      >
        <div className={`p-2 rounded-lg flex-shrink-0 ${color}`}>
          <Icon className="w-4 h-4 text-white" />
        </div>
        <div className="flex-1 min-w-0">

          <p className="font-semibold text-sm text-white">{agent}</p>

          <p className="text-xs text-slate-400 mt-1 line-clamp-1">
            {analysis}
          </p>
        </div>

        <ChevronDown
          className={`w-4 h-4 text-slate-400 flex-shrink-0 transition-transform ${
            expanded ? 'rotate-180' : ''
          }`}
        />
      </button>

      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}

            exit={{ opacity: 0, height: 0 }}
            className="border-t border-white/10 px-4 py-3 bg-white/3"
          >
            <p className="text-xs text-slate-300 leading-relaxed break-words">
              {analysis}
            </p>
                  {/* this for the grad cam image generation*/}
                  {agentData.gradcam_image && (
                    <div className="mt-3">
                      <p className="text-xxs text-slate-400 uppercase mb-2">Grad-CAM</p>
                      <img
                        src={agentData.gradcam_image}
                        alt="Grad-CAM"
                        className="w-full rounded-md border border-white/10"
                      />
                    </div>
                  )}
                  {/* this for the shap image generation if needed only */}

                  {agentData.shap_image && (
                    <div className="mt-3">
                      <p className="text-xxs text-slate-400 uppercase mb-2">Clinical Contributions</p>
                      <img
                        src={agentData.shap_image}
                        alt="SHAP"

                        className="w-full rounded-md border border-white/10"
                      />

                    </div>
                  )}
            {/* ROI of mri scan analysis if present/needed*/}
            {agentData.rois && agentData.rois.length > 0 && (
              <div className="mt-3">

                <p className="text-xxs text-slate-400 uppercase mb-2">ROIs</p>
                <ul className="text-xs text-slate-200 space-y-1">
                  {agentData.rois.map((r, i) => (
                    <li key={i} className="flex justify-between">
                      <span>{r.region}</span>
                      {r.confidence !== undefined && (

                        <span className="text-slate-400">{(r.confidence*100).toFixed(0)}%</span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            )}




            {agentData.associated_rois && agentData.associated_rois.length > 0 && (


              <div className="mt-3">
                <p className="text-xxs text-slate-400 uppercase mb-2">Correlated ROIs</p>

                <ul className="text-xs text-slate-200 space-y-1">
                  {agentData.associated_rois.map((a, i) => (
                    <li key={i}>
                      <span className="font-medium">{a.feature}:</span> {a.rois.join(', ')}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {agentData.linked_rois && (
              <div className="mt-3">
                <p className="text-xxs text-slate-400 uppercase mb-2">Linked ROIs</p>
                <p className="text-xs text-slate-200">{agentData.linked_rois.join(', ')}</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}



export default function AgentPanel({ reasoning, loading }) {
  if (loading && !reasoning) {
    return (
      <div className="glass-effect p-6 h-full flex flex-col">
        <h2 className="text-xl font-bold text-cyan-400 mb-6 flex items-center gap-2">
          <MessageCircle className="w-5 h-5" />
          Agent Discussion
        </h2>
        <div className="flex-1 flex flex-col items-center justify-center">

          <div className="space-y-2 w-full">
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}


                className="h-16 bg-white/5 rounded-lg animate-pulse border border-white/10"
              ></div>
            ))}
          </div>
          <p className="text-slate-400 text-sm mt-4 text-center">
            Agents are analyzing the data...
          </p>
        </div>
      </div>
    )
  }

  if (!reasoning) {
    return (
      <div className="glass-effect p-6 h-full flex flex-col items-center justify-center text-center">
        <MessageCircle className="w-12 h-12 text-slate-600 mb-3" />
        <p className="text-slate-400 text-sm">Agent reasoning will appear here</p>
      </div>
    )
  }

  const agents = [
    {
      key: 'gradcam',
      icon: Microscope,
      color: 'bg-blue-500/20',
      label: 'Neuroimaging',
    },


    {

      key: 'shap',
      icon: Heart,
      color: 'bg-pink-500/20',
      label: 'Clinical Features',
    },
    {


      key: 'clinical_plausibility',
      icon: CheckCircle,
      color: 'bg-green-500/20',
      label: 'Plausibility Check',
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-effect p-6 h-full flex flex-col"
    >
      <h2 className="text-xl font-bold text-cyan-400 mb-6 flex items-center gap-2">
        <MessageCircle className="w-5 h-5" />
        Agent Consensus
      </h2>

      {/* Agent Cards */}
      <div className="flex-1 overflow-y-auto pr-2 mb-6 space-y-2">
        {agents.map(({ key, icon, color, label }) => {
          const agentData = reasoning[key]
          if (!agentData) return null

          return (
            <AgentCard
              key={key}
              agentData={agentData}
              icon={icon}
              color={color}
            />
          )
        })}
      </div>





      {/* Final Consensus Summary */}
      {reasoning.final_consensus && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-auto p-4 rounded-lg bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30"
        >
          <p className="text-xs font-semibold text-cyan-300 uppercase mb-2">
            ✓ Final Consensus Reached
          </p>
          <p className="text-xs text-slate-200 leading-relaxed">
            All agents have reached agreement on the diagnosis. The multi-agent framework
            has synthesized neuroimaging, clinical, and genetic evidence into a unified
            clinical assessment.
          </p>
        </motion.div>
      )}
    </motion.div>
  )
}
