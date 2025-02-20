import { useState, useCallback } from 'react';
import { FileUploadState } from '../types/FileUpload';

interface AnalysisResult {
  duplicate_groups: Array<{
    files: string[];
  }>;
  similar_images: Array<{
    image1: string;
    image2: string;
    similarity_score: number;
    inliers: number;
  }>;
  top_pairs: Array<{
    image1: string;
    image2: string;
    inliers: number;
    clip_score: number;
  }>;
  total_images: number;
  processing_time: number;
  progress: number;
}

export const allowedTypes = [
  'image/jpeg',
  'image/png',
  'image/gif',
  'image/bmp',
  'image/tiff',
  'application/pdf'
];

export function useFileUpload(onAnalysisComplete?: (results: AnalysisResult) => void) {
  const [state, setState] = useState<FileUploadState>({
    file: null,
    files: [],
    progress: 0,
    status: 'idle',
    sessionId: null,
  });

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => allowedTypes.includes(file.type));
    
    if (validFiles.length > 0) {
      setState(prev => ({ ...prev, files: validFiles, status: 'idle', error: undefined }));
    } else {
      setState(prev => ({ 
        ...prev, 
        error: 'Invalid file type. Please upload images (JPEG, PNG, GIF, BMP, TIFF) or PDF files only.' 
      }));
    }
  }, []);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    const validFiles = files.filter(file => allowedTypes.includes(file.type));
    
    if (validFiles.length > 0) {
      setState(prev => ({ ...prev, files: validFiles, status: 'idle', error: undefined }));
    } else {
      setState(prev => ({ 
        ...prev, 
        error: 'Invalid file type. Please upload images (JPEG, PNG, GIF, BMP, TIFF) or PDF files only.' 
      }));
    }
  }, []);

  const handleUpload = useCallback(async () => {
    if (!state.files?.length) return;

    setState(prev => ({ ...prev, status: 'uploading', progress: 0 }));

    try {
      // Create FormData with files
      const formData = new FormData();
      state.files.forEach(file => {
        formData.append('files', file);  // Match the FastAPI parameter name
      });

      // Start the upload
      const uploadResponse = await fetch('http://localhost:8001/upload', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        const errorData = await uploadResponse.text();
        throw new Error(`Upload failed: ${errorData}`);
      }

      const uploadData = await uploadResponse.json();
      const sessionId = uploadData.session_id;

      setState(prev => ({
        ...prev,
        sessionId
      }));

      // Start listening for progress updates
      const eventSource = new EventSource('http://localhost:8001/progress');

      eventSource.onmessage = (event: MessageEvent) => {
        const data = JSON.parse(event.data);
        if (data.progress) {
          setState(prev => ({
            ...prev,
            progress: data.progress
          }));
        }
      };

      eventSource.addEventListener('progress', (event: MessageEvent) => {
        const data = JSON.parse(event.data);
        setState(prev => ({
          ...prev,
          progress: data.progress
        }));
      });

      eventSource.addEventListener('complete', (event: MessageEvent) => {
        const results = JSON.parse(event.data);
        setState(prev => ({
          ...prev,
          status: 'success',
          progress: 100,
        }));
        if (onAnalysisComplete) {
          onAnalysisComplete(results);
        }
        eventSource.close();
      });

      eventSource.addEventListener('error', (event: MessageEvent) => {
        const data = JSON.parse(event.data);
        setState(prev => ({
          ...prev,
          status: 'error',
          error: data.error || 'An error occurred during analysis',
        }));
        eventSource.close();
      });

      eventSource.onerror = () => {
        setState(prev => ({
          ...prev,
          status: 'error',
          error: 'Connection to server lost',
        }));
        eventSource.close();
      };

    } catch (error) {
      setState(prev => ({
        ...prev,
        status: 'error',
        error: error instanceof Error ? error.message : 'An error occurred during analysis',
      }));
    }
  }, [state.files, onAnalysisComplete]);

  const handleClear = useCallback(async () => {
    try {
      if (state.sessionId) {
        // Clean up session-specific files
        await fetch(`http://localhost:8001/cleanup/${state.sessionId}`, {
          method: 'POST'
        });
      }
      
      setState({
        file: null,
        files: [],
        progress: 0,
        status: 'idle',
        sessionId: null
      });
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  }, [state.sessionId]);

  const handleFileDelete = useCallback((index: number) => {
    if (index === -1) {
      // Special case: clear all files
      handleClear();
    } else {
      setState(prev => ({
        ...prev,
        files: prev.files.filter((_, i) => i !== index)
      }));
    }
  }, [handleClear]);

  return {
    state,
    handlers: {
      handleDragOver,
      handleDrop,
      handleFileSelect,
      handleUpload,
      handleClear,
      handleFileDelete
    }
  };
}