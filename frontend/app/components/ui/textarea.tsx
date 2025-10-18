'use client'

import React from 'react'

interface TextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  children?: React.ReactNode
}

export function Textarea({ className = '', children, ...props }: TextareaProps) {
  const baseClasses = 'flex min-h-[80px] w-full rounded-md border border-night-600 bg-night-800 px-3 py-2 text-sm text-night-100 placeholder:text-night-400 focus:outline-none focus:ring-2 focus:ring-neon-500 focus:border-transparent disabled:cursor-not-allowed disabled:opacity-50'
  
  const classes = `${baseClasses} ${className}`
  
  return (
    <textarea className={classes} {...props}>
      {children}
    </textarea>
  )
}
