import React from 'react';
import { Upload, Loader2, X, Folder, Trash2, RefreshCw } from 'lucide-react';
import { FileUploadState } from '../types/FileUpload';

// Extend HTMLInputElement to include directory upload attributes
declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    webkitdirectory?: string;
    directory?: string;
  }
}

interface AnalysisResult {
  imagePair: [string, string];
  inliers: number;
  clipScore: number;
}

interface FileUploadProps {
  state: FileUploadState;
  onDragOver: (e: React.DragEvent) => void;
  onDrop: (e: React.DragEvent) => void;
  onFileSelect: (e: React.ChangeEvent<HTMLInputElement>) => void;
  onUpload: () => void;
  onFileDelete: (index: number) => void;
}

export function FileUpload({
  state,
  onDragOver,
  onDrop,
  onFileSelect,
  onUpload,
  onFileDelete
}: FileUploadProps) {
  const handleCleanup = async () => {
    try {
      const response = await fetch('http://localhost:8001/cleanup', {
        method: 'POST'
      });
      if (!response.ok) {
        throw new Error('Cleanup failed');
      }
      // Clear the file list after successful cleanup
      onFileDelete(-1); // Special value to clear all files
    } catch (error) {
      console.error('Cleanup error:', error);
    }
  };

  return (
    <section className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold">File Upload</h2>
        <button
          onClick={handleCleanup}
          className="p-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg flex items-center"
          title="Clean up uploaded files"
        >
          <RefreshCw className="w-5 h-5 mr-1" />
          Clean Up
        </button>
      </div>
      
      <div
        onDragOver={onDragOver}
        onDrop={onDrop}
        className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition-colors"
      >
        {/* Input for multiple files */}
        <input
          type="file"
          id="file-upload"
          className="hidden"
          onChange={onFileSelect}
          accept=".png,.jpg,.jpeg,.bmp,.tiff,.pdf"
          multiple
        />
        {/* Input for folder */}
        <input
          type="file"
          id="folder-upload"
          className="hidden"
          onChange={onFileSelect}
          accept=".png,.jpg,.jpeg,.bmp,.tiff,.pdf"
          webkitdirectory=""
          directory=""
        />
        
        <div className="flex flex-col items-center">
          <Upload className="w-12 h-12 text-gray-400 mb-4" />
          <p className="text-lg text-gray-600 mb-2">
            Drag and drop files here or use buttons below
          </p>
          <p className="text-sm text-gray-500 mb-4">
            Supported formats: PNG, JPG, JPEG, BMP, TIFF, PDF
          </p>
          
          <div className="flex gap-4">
            <label
              htmlFor="file-upload"
              className="cursor-pointer px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors flex items-center"
            >
              <Upload className="w-4 h-4 mr-2" />
              Select Files
            </label>
            <label
              htmlFor="folder-upload"
              className="cursor-pointer px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors flex items-center"
            >
              <Folder className="w-4 h-4 mr-2" />
              Select Folder
            </label>
          </div>
        </div>
      </div>

      {state.error && (
        <div className="mt-4 p-4 bg-red-50 text-red-700 rounded-lg flex items-center">
          <X className="w-5 h-5 mr-2" />
          {state.error}
        </div>
      )}

      {state.files && state.files.length > 0 && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium">{state.files.length} files selected</p>
              <p className="text-sm text-gray-500">
                Total size: {(state.files.reduce((acc, file) => acc + file.size, 0) / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
            <button
              onClick={onUpload}
              disabled={state.status === 'uploading'}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
            >
              {state.status === 'uploading' ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing... {state.progress}%
                </>
              ) : (
                <>
                  <Upload className="w-4 h-4 mr-2" />
                  Analyze
                </>
              )}
            </button>
          </div>
          
          {state.status === 'uploading' && (
            <div className="mt-4">
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-2 bg-blue-600 rounded-full transition-all duration-300"
                  style={{ width: `${state.progress}%` }}
                />
              </div>
            </div>
          )}

          {/* File List */}
          <div className="mt-4 max-h-40 overflow-y-auto">
            <div className="space-y-2">
              {state.files.map((file, index) => (
                <div key={index} className="flex items-center justify-between py-1 px-2 bg-white rounded hover:bg-gray-50">
                  <span className="text-sm truncate flex-1">{file.name}</span>
                  <div className="flex items-center gap-2 ml-2">
                    <span className="text-xs text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </span>
                    <button
                      onClick={() => onFileDelete(index)}
                      className="p-1 text-gray-400 hover:text-red-500 rounded transition-colors"
                      title="Delete file"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}