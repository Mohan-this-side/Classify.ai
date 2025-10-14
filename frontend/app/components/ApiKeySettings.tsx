'use client'

import React, { useState, useEffect } from 'react'
import { Key, Eye, EyeOff, Check, AlertCircle, Info } from 'lucide-react'
import { toast } from 'react-hot-toast'

interface ApiKeySettingsProps {
  onApiKeyChange: (apiKey: string) => void
  initialApiKey?: string
  disabled?: boolean
}

export default function ApiKeySettings({ 
  onApiKeyChange, 
  initialApiKey = '', 
  disabled = false 
}: ApiKeySettingsProps) {
  const [apiKey, setApiKey] = useState(initialApiKey)
  const [showApiKey, setShowApiKey] = useState(false)
  const [isValid, setIsValid] = useState<boolean | null>(null)
  const [isValidating, setIsValidating] = useState(false)

  // Validate API key format
  const validateApiKey = (key: string): boolean => {
    // Google Gemini API key format validation
    // Should start with "AIza" and be 39 characters long
    const geminiKeyPattern = /^AIza[0-9A-Za-z-_]{35}$/
    return geminiKeyPattern.test(key)
  }

  // Test API key by making a simple request
  const testApiKey = async (key: string): Promise<boolean> => {
    if (!key) return false
    
    try {
      setIsValidating(true)
      
      // Test the API key by making a simple request to the backend
      const response = await fetch('http://localhost:8000/api/test-gemini-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ api_key: key })
      })
      
      if (response.ok) {
        const result = await response.json()
        return result.valid === true
      }
      
      return false
    } catch (error) {
      console.error('Error testing API key:', error)
      return false
    } finally {
      setIsValidating(false)
    }
  }

  // Handle API key input change
  const handleApiKeyChange = async (value: string) => {
    setApiKey(value)
    onApiKeyChange(value)
    
    if (value.length === 0) {
      setIsValid(null)
      return
    }
    
    // Basic format validation
    const formatValid = validateApiKey(value)
    if (!formatValid) {
      setIsValid(false)
      return
    }
    
    // Test the API key if it's properly formatted
    const testResult = await testApiKey(value)
    setIsValid(testResult)
    
    if (testResult) {
      toast.success('API key validated successfully!')
    } else if (value.length === 39) {
      toast.error('Invalid API key. Please check your key and try again.')
    }
  }

  // Load API key from localStorage on mount
  useEffect(() => {
    const savedApiKey = localStorage.getItem('gemini_api_key')
    if (savedApiKey) {
      setApiKey(savedApiKey)
      onApiKeyChange(savedApiKey)
    }
  }, [onApiKeyChange])

  // Save API key to localStorage when it changes
  useEffect(() => {
    if (apiKey && isValid) {
      localStorage.setItem('gemini_api_key', apiKey)
    }
  }, [apiKey, isValid])

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <Key className="w-5 h-5 text-neon-400" />
        <h3 className="text-lg font-semibold text-white">Gemini API Configuration</h3>
      </div>
      
      <div className="space-y-3">
        <div>
          <label htmlFor="api-key" className="block text-sm font-medium text-gray-300 mb-2">
            Google Gemini API Key
          </label>
          <div className="relative">
            <input
              id="api-key"
              type={showApiKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => handleApiKeyChange(e.target.value)}
              disabled={disabled}
              placeholder="Enter your Gemini API key (AIza...)"
              className={`w-full px-4 py-3 bg-gray-800/50 border rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 transition-all duration-200 ${
                isValid === true 
                  ? 'border-green-500 focus:ring-green-500/50' 
                  : isValid === false 
                  ? 'border-red-500 focus:ring-red-500/50'
                  : 'border-gray-600 focus:ring-neon-500/50'
              } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            />
            <button
              type="button"
              onClick={() => setShowApiKey(!showApiKey)}
              disabled={disabled}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-300 transition-colors"
            >
              {showApiKey ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
            </button>
            
            {/* Validation indicator */}
            {apiKey && (
              <div className="absolute right-12 top-1/2 transform -translate-y-1/2">
                {isValidating ? (
                  <div className="w-5 h-5 border-2 border-neon-400 border-t-transparent rounded-full animate-spin" />
                ) : isValid === true ? (
                  <Check className="w-5 h-5 text-green-400" />
                ) : isValid === false ? (
                  <AlertCircle className="w-5 h-5 text-red-400" />
                ) : null}
              </div>
            )}
          </div>
        </div>
        
        {/* API Key Status */}
        {apiKey && (
          <div className={`p-3 rounded-lg text-sm ${
            isValid === true 
              ? 'bg-green-500/10 border border-green-500/20 text-green-400'
              : isValid === false
              ? 'bg-red-500/10 border border-red-500/20 text-red-400'
              : 'bg-yellow-500/10 border border-yellow-500/20 text-yellow-400'
          }`}>
            <div className="flex items-center gap-2">
              {isValidating ? (
                <>
                  <div className="w-4 h-4 border-2 border-neon-400 border-t-transparent rounded-full animate-spin" />
                  <span>Validating API key...</span>
                </>
              ) : isValid === true ? (
                <>
                  <Check className="w-4 h-4" />
                  <span>API key is valid and ready to use</span>
                </>
              ) : isValid === false ? (
                <>
                  <AlertCircle className="w-4 h-4" />
                  <span>Invalid API key. Please check your key and try again.</span>
                </>
              ) : (
                <>
                  <Info className="w-4 h-4" />
                  <span>Enter a valid Gemini API key to enable AI features</span>
                </>
              )}
            </div>
          </div>
        )}
        
        {/* Help text */}
        <div className="text-sm text-gray-400 space-y-2">
          <p>
            <strong>How to get your Gemini API key:</strong>
          </p>
          <ol className="list-decimal list-inside space-y-1 ml-4">
            <li>Visit the <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-neon-400 hover:text-neon-300 underline">Google AI Studio</a></li>
            <li>Sign in with your Google account</li>
            <li>Click "Create API Key"</li>
            <li>Copy the generated key and paste it above</li>
          </ol>
          <p className="text-xs text-gray-500 mt-2">
            Your API key is stored locally in your browser and only sent to our secure backend for processing.
          </p>
        </div>
      </div>
    </div>
  )
}
