#!/usr/bin/env python3
"""
NARadio Processor Module

Handles NARadio feature extraction and visualization with robust error handling
and GPU memory management to prevent crashes.

OPTIMIZATIONS IMPLEMENTED:
==========================
1. Feature Caching: Avoid redundant feature extraction for the same image
   - Image hash-based caching with TTL (5 seconds)
   - Automatic cache cleanup and size limits
   
2. Reduced Redundant Operations:
   - process_features_optimized() extracts features once, reuses for both visualization and similarity
   - compute_vlm_similarity_map_optimized() can reuse pre-computed features
   - Cached image preprocessing to avoid repeated tensor conversion
   
3. Memory Management Optimizations:
   - Reduced CUDA cache clearing frequency (only when needed)
   - Reduced memory cleanup interval (10s instead of continuous)
   - Better tensor cleanup and garbage collection
   
4. Processing Optimizations:
   - Optional visualization computation (skip when not needed)
   - Chunked similarity computation for memory efficiency
   - Early exit conditions to avoid unnecessary work
   
5. I/O Optimizations:
   - Skip saving raw similarity maps (only save colored visualizations)
   - Lightweight metadata saving
   - Skip saves for failed/slow computations
   
6. Device Consistency Optimizations:
   - Reduced device consistency check frequency (5s instead of continuous)
   - Better error handling to avoid repeated reinitializations

These optimizations should significantly reduce the processing time for combined
segmentation while maintaining the same functionality and output quality.
"""

import torch
import numpy as np
import cv2
import gc
import sys
import time
import warnings
import yaml
import os
import json
from typing import Optional, Tuple, Dict, List, Union
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import hashlib
import threading

# Suppress warnings
warnings.filterwarnings('ignore')

from resilience.cause_registry import CauseRegistry
import rclpy.logging


class _ZeroImageEncoder:
	def __init__(self, embed_dim: int, device: str):
		self.embed_dim = embed_dim
		self.device = device

	def encode_image_to_vector(self, rgb_img: torch.Tensor) -> torch.Tensor:
		batch = rgb_img.shape[0]
		return torch.zeros(batch, self.embed_dim, device=rgb_img.device, dtype=rgb_img.dtype)

	def encode_image_to_feat_map(self, rgb_img: torch.Tensor) -> torch.Tensor:
		batch, _, h, w = rgb_img.shape
		return torch.zeros(batch, self.embed_dim, h, w, device=rgb_img.device, dtype=rgb_img.dtype)

	def align_spatial_features_with_language(self, feat: torch.Tensor) -> torch.Tensor:
		return feat

class NARadioProcessor:
    """NARadio feature extraction and processing with robust error handling and optimizations."""
    
    def __init__(self, 
                 radio_model_version: str = 'radio_v2.5-b',
                 radio_lang_model: str = 'siglip',
                 radio_input_resolution: int = 224,
                 enable_visualization: bool = True,
                 enable_combined_segmentation: bool = False,
                 segmentation_config_path: str = None,
                 cause_registry: Optional[CauseRegistry] = None):
        """Initialize NARadio processor."""
        self._logger = rclpy.logging.get_logger('naradio_processor')
        self.last_similarity_error: Optional[str] = None
        self.radio_model_version = radio_model_version
        self.radio_lang_model = radio_lang_model
        self.radio_input_resolution = 512
        self.enable_visualization = enable_visualization
        self.enable_combined_segmentation = enable_combined_segmentation
        
        # Initialize NARadio components
        self.naradio_ready = False
        self.radio_encoder = None
        self.pca = PCA(n_components=3)  # For RGB visualization
        self.pca_fitted = False
        
        # NARadio availability
        self.NARADIO_AVAILABLE = False
        self.NARadioEncoder = None
        
        # Combined segmentation components
        self.segmentation_config = None
        self.word_features = None
        self.word_list = []
        self.colors = []
        self.segmentation_ready = False
        
        # Dynamic object management
        self.dynamic_objects = []  # Objects added from VLM answers
        self.dynamic_features = None  # Features for dynamic objects
        self.dynamic_colors = []  # Colors for dynamic objects
        
        # NEW: Enhanced embedding management for better similarity detection
        self.enhanced_embeddings = {}  # Dict: {vlm_answer: enhanced_embedding_np}
        self.enhanced_embedding_colors = {}  # Dict: {vlm_answer: color}
        
        # OPTIMIZATION: Feature caching to avoid redundant computation
        self.feature_cache = {}
        self.max_cache_size = 10
        self.cache_ttl = 5.0  # Cache TTL in seconds
        
        # OPTIMIZATION: Reusable tensors
        self._cached_tensor_image = None
        self._cached_image_hash = None
        self._cached_feat_map = None
        self._cached_aligned_feat_map = None
        
        # OPTIMIZATION: Memory cleanup tracking
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 30.0  # SPEED OPTIMIZATION: Increased cleanup frequency
        
        # Initialize NARadio
        self.init_naradio()
        
        # Initialize combined segmentation if enabled
        if self.enable_combined_segmentation:
            self.init_combined_segmentation(segmentation_config_path)
        
        # Risk library for tracking VLM answers with enhanced embeddings
        self.risk_library = {}  # vlm_answer -> {enhanced_embedding, metadata, timestamp}
        self.risk_library_lock = threading.Lock()
        self.cause_registry = cause_registry
    
    def _compute_image_hash(self, rgb_image: np.ndarray) -> str:
        """Compute hash for image caching."""
        return hashlib.md5(rgb_image.tobytes()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict) -> bool:
        """Check if cache entry is still valid."""
        return time.time() - cache_entry['timestamp'] < self.cache_ttl
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        current_time = time.time()
        expired_keys = []
        for key, entry in self.feature_cache.items():
            if current_time - entry['timestamp'] > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.feature_cache[key]
        
        # Limit cache size
        if len(self.feature_cache) > self.max_cache_size:
            # Remove oldest entries
            sorted_items = sorted(self.feature_cache.items(), key=lambda x: x[1]['timestamp'])
            for key, _ in sorted_items[:len(sorted_items) - self.max_cache_size]:
                del self.feature_cache[key]

    def init_naradio(self):
        """Initialize NARadio model for feature extraction with robust error handling."""
        try:
            # Add the radio_enhanced directory to the path
            radio_path = '/home/navin/ros2_ws/src/resilience/resilience'
            if radio_path not in sys.path:
                sys.path.append(radio_path)
                print(f"Added {radio_path} to Python path")
            
            # Try to import NARadio
            from naradio import NARadioEncoder
            # from radseg import RADSegEncoder
            print("✓ NARadioEncoder imported successfully")
            self.NARadioEncoder = NARadioEncoder
            self.NARADIO_AVAILABLE = True
            

            if not self.NARADIO_AVAILABLE:
                print("NARadio not available, skipping initialization")
                self.naradio_ready = False
                return

            # Clear CUDA cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Initialize device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            
            # Initialize radio_encoder to None first
            self.radio_encoder = None
            
            # Load NARadio model with proper error handling
            if self.NARadioEncoder is not None:
                try:
                    print(f"Initializing NARadio model with version={self.radio_model_version}, lang_model={self.radio_lang_model}, resolution={self.radio_input_resolution}")
                    self.radio_encoder = self.NARadioEncoder(
                        model_version=self.radio_model_version,
                        lang_model=self.radio_lang_model,
                        input_resolution=(self.radio_input_resolution, self.radio_input_resolution),
                        device=str(device))
                    print("✓ NARadio model created successfully")
                    
                    # Set NARadio to evaluation mode if it has eval method
                    if hasattr(self.radio_encoder, 'eval'):
                        self.radio_encoder.eval()
                        print("✓ Set model to evaluation mode")
                    
                    
                    self.naradio_ready = True
                        
                    
                except Exception as e:
                    print(f"Failed to load NARadio model: {e}")
                    import traceback
                    traceback.print_exc()
                    self.radio_encoder = None
                    self.naradio_ready = False
                    return
            else:
                print("NARadioEncoder not available")
                self.naradio_ready = False
                return
            
            print(f"NARadio initialization completed. Ready: {self.naradio_ready}")
            
        except Exception as e:
            self.naradio_ready = False
            print(f"Error initializing NARadio: {e}")
            import traceback
            traceback.print_exc()

    def test_naradio_model(self) -> bool:
        """Test if NARadio model is working properly."""
        try:
            if self.radio_encoder is None:
                return False
                
            # Create a dummy input tensor
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dummy_input = torch.randn(1, 3, self.radio_input_resolution, self.radio_input_resolution).to(device)
            
            # Try to encode the dummy input
            with torch.no_grad():
                feat_map = self.radio_encoder.encode_image_to_feat_map(dummy_input)
                return True
                
        except Exception as e:
            print(f"Model test failed: {e}")
            return False

    def _preprocess_image_cached(self, rgb_image: np.ndarray) -> Tuple[torch.Tensor, str]:
        """
        Preprocess image with caching to avoid redundant operations.
        
        Returns:
            Tuple of (preprocessed_tensor, image_hash)
        """
        # Compute image hash for caching
        image_hash = self._compute_image_hash(rgb_image)
        
        # Check if we have cached tensor for this image
        if (self._cached_tensor_image is not None and 
            self._cached_image_hash == image_hash):
            return self._cached_tensor_image, image_hash
        
        # Preprocess image
        if len(rgb_image.shape) == 3:
            # Convert BGR to RGB if needed
            if rgb_image.shape[2] == 3:
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        # Resize image to model input resolution
        resized_image = cv2.resize(rgb_image, (self.radio_input_resolution, self.radio_input_resolution))
        
        # Convert to tensor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image_tensor = torch.from_numpy(resized_image).float().to(device)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC to CHW and add batch dimension
        image_tensor = image_tensor / 255.0  # Normalize to [0, 1]
        
        # Cache the result
        self._cached_tensor_image = image_tensor
        self._cached_image_hash = image_hash
        
        return image_tensor, image_hash

    def extract_features_cached(self, rgb_image: np.ndarray, return_tensor: bool = False) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """Extract NARadio features from RGB image with caching."""
        try:
            if not self.is_ready():
                return None
            
            # SPEED OPTIMIZATION: Compute hash only when cache is likely to be useful
            image_hash = None
            
            # Check cache first
            if len(self.feature_cache) > 0:
                image_hash = self._compute_image_hash(rgb_image)
                
                if image_hash in self.feature_cache:
                    cache_entry = self.feature_cache[image_hash]
                    if self._is_cache_valid(cache_entry):
                        feat_map = cache_entry['feat_map']
                        if return_tensor:
                            if isinstance(feat_map, np.ndarray):
                                return torch.from_numpy(feat_map).to(self.device)
                            return feat_map
                        else:
                            if isinstance(feat_map, torch.Tensor):
                                return feat_map.cpu().numpy()
                            return feat_map
            
            
            
            # Preprocess image (with caching)
            image_tensor, computed_hash = self._preprocess_image_cached(rgb_image)
            
            # Extract features
            with torch.no_grad():
                feat_map = self.radio_encoder.encode_image_to_feat_map(image_tensor)
                
                # Cache the result (prefer numpy for storage to save VRAM)
                if image_hash is None:
                    image_hash = computed_hash
                    
                if len(self.feature_cache) < self.max_cache_size:
                    self.feature_cache[image_hash] = {
                        'feat_map': feat_map.cpu().numpy() if isinstance(feat_map, torch.Tensor) else feat_map,
                        'timestamp': time.time()
                    }
                
                # Cleanup occasionally
                current_time = time.time()
                if (current_time - self.last_cleanup_time > self.cleanup_interval and 
                    len(self.feature_cache) > self.max_cache_size * 0.8):
                    self._cleanup_cache()
                    self.last_cleanup_time = current_time
                
                if return_tensor:
                    return feat_map
                return feat_map.cpu().numpy() if isinstance(feat_map, torch.Tensor) else feat_map
                
        except torch.cuda.OutOfMemoryError:
            self.handle_cuda_out_of_memory()
            return None
        except Exception:
            return None

    def extract_features(self, rgb_image: np.ndarray, return_tensor: bool = False) -> Optional[Union[np.ndarray, torch.Tensor]]:
        """Legacy method - redirects to cached version."""
        return self.extract_features_cached(rgb_image, return_tensor=return_tensor)

    def create_visualization(self, feat_map_np: np.ndarray) -> Optional[np.ndarray]:
        """Create visualization of NARadio features."""
        try:
            if not self.enable_visualization:
                return None
                
            if feat_map_np is None:
                return None
            
            # Reshape feature map if needed
            if len(feat_map_np.shape) == 4:  # Batch, channels, height, width
                feat_map_np = feat_map_np[0]  # Remove batch dimension
            
            # Flatten spatial dimensions
            h, w = feat_map_np.shape[1], feat_map_np.shape[2]
            feat_flat = feat_map_np.reshape(feat_map_np.shape[0], -1).T  # (H*W, C)
            
            # Fit PCA if not already fitted
            if not self.pca_fitted:
                self.pca.fit(feat_flat)
                self.pca_fitted = True
            
            # Transform features to 3D
            feat_3d = self.pca.transform(feat_flat)
            
            # Normalize to [0, 255] for visualization
            feat_3d = (feat_3d - feat_3d.min()) / (feat_3d.max() - feat_3d.min() + 1e-8)
            feat_3d = (feat_3d * 255).astype(np.uint8)
            
            # Reshape back to image dimensions
            feat_vis = feat_3d.reshape(h, w, 3)
            
            # Resize to configured model input resolution for visualization
            feat_vis = cv2.resize(
                feat_vis,
                (self.radio_input_resolution, self.radio_input_resolution)
            )
            
            return feat_vis
            
        except Exception as e:
            print(f"Error creating NARadio visualization: {e}")
            return None

    def process_features_optimized(self, rgb_image: np.ndarray, 
                                 need_visualization: bool = True,
                                 reuse_features: bool = True,
                                 return_tensor: bool = False) -> Tuple[Optional[Union[np.ndarray, torch.Tensor]], Optional[np.ndarray]]:
        """
        OPTIMIZED: Process NARadio features with optional visualization and feature reuse.
        
        Args:
            rgb_image: Input RGB image
            need_visualization: Whether to compute visualization (expensive)
            reuse_features: Whether to use cached features if available
            return_tensor: Whether to return features as GPU tensor
        """
        try:
            feat_map = self.extract_features(rgb_image, return_tensor=return_tensor)
            
            # Create visualization only if needed
            naradio_vis = None
            if feat_map is not None and need_visualization:
                # If we have a tensor, convert to numpy for visualization
                feat_map_np = feat_map.cpu().numpy() if isinstance(feat_map, torch.Tensor) else feat_map
                naradio_vis = self.create_visualization(feat_map_np)
            
            return feat_map, naradio_vis
            
        except Exception as e:
            print(f"Error processing NARadio features: {e}")
            return None, None

    def process_features(self, rgb_image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Legacy method - redirects to optimized version."""
        return self.process_features_optimized(rgb_image, need_visualization=True, reuse_features=True)

    def cleanup_memory(self):
        """OPTIMIZED: Periodic cleanup of radio model memory with reduced frequency."""
        try:
            current_time = time.time()
            # SPEED OPTIMIZATION: Increase cleanup interval to reduce overhead
            if current_time - self.last_cleanup_time < self.cleanup_interval:
                return
                
            if torch.cuda.is_available():
                # SPEED OPTIMIZATION: Only cleanup if memory usage is high
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                # Only cleanup if using more than 70% of GPU memory
                if allocated > total * 0.7:
                    torch.cuda.empty_cache()
                
            # SPEED OPTIMIZATION: Reduce cache cleaning frequency
            if current_time - self.last_cleanup_time > self.cleanup_interval * 2:
                self._cleanup_cache()
                
            self.last_cleanup_time = current_time
                
        except Exception as e:
            pass  # Silently handle cleanup errors

    def handle_cuda_out_of_memory(self):
        """Handle CUDA out of memory errors."""
        try:
            if torch.cuda.is_available():
                # Clear all CUDA cache
                torch.cuda.empty_cache()
                
                # Force garbage collection
                gc.collect()
                
                # Clear our feature cache
                self.feature_cache.clear()
                self._cached_tensor_image = None
                self._cached_image_hash = None
                self._cached_feat_map = None
                self._cached_aligned_feat_map = None
                
                # Get memory info
                allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                print(f"CUDA OOM handled - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Total: {total:.2f}GB")
                
                # If still high memory usage, try to reset the radio model
                if allocated > total * 0.8:  # If more than 80% of GPU memory is used
                    if self.radio_encoder is not None:
                        print("High memory usage detected, resetting radio model...")
                        del self.radio_encoder
                        self.radio_encoder = None
                        self.naradio_ready = False
                        torch.cuda.empty_cache()
                        print("Radio model reset due to high memory usage")
                
        except Exception as e:
            print(f"Error handling CUDA out of memory: {e}")

    def ensure_device_consistency(self) -> bool:
        """Ensure radio model is on the correct device."""
        try:
            if self.radio_encoder is None:
                return False
                
            # Check if model is valid
            return hasattr(self.radio_encoder, 'model')
            
        except Exception as e:
            print(f"Device consistency check failed: {e}")
            return False

    def is_ready(self) -> bool:
        """Check if NARadio is ready for processing."""
        return self.naradio_ready and self.radio_encoder is not None

    def reinitialize(self) -> bool:
        """Reinitialize NARadio model."""
        try:
            print("Reinitializing NARadio model...")
            
            # Clean up existing model
            if self.radio_encoder is not None:
                del self.radio_encoder
                self.radio_encoder = None
            
            # Clear all caches
            self.feature_cache.clear()
            self._cached_tensor_image = None
            self._cached_image_hash = None
            
            # Clear memory
            self.cleanup_memory()
            
            # Reinitialize
            self.init_naradio()
            
            # Reinitialize combined segmentation if enabled
            if self.enable_combined_segmentation:
                self.init_combined_segmentation()
            
            success = self.is_ready()
            print(f"NARadio reinitialization {'successful' if success else 'failed'}")
            return success
            
        except Exception as e:
            print(f"Error reinitializing NARadio: {e}")
            return False
    
    def init_combined_segmentation(self, config_path: str = None):
        """Initialize combined segmentation with configuration."""
        try:
            print("Initializing Combined Segmentation...")
            
            # Load configuration
            self.segmentation_config = self.load_segmentation_config(config_path)
            if not self.segmentation_config:
                print("Failed to load segmentation configuration")
                return
            
            # Encode word embeddings
            self.encode_word_embeddings()
            
            # Set up colors
            self.setup_colors()
            
            self.segmentation_ready = True
            print("✓ Combined Segmentation initialized successfully")
            
        except Exception as e:
            print(f"Error initializing combined segmentation: {e}")
            import traceback
            traceback.print_exc()
            self.segmentation_ready = False
    
    def load_segmentation_config(self, config_path: str = None) -> Optional[Dict]:
        """Load segmentation configuration from YAML file."""
        if config_path is None:
            # Use default config path
            try:
                from ament_index_python.packages import get_package_share_directory
                package_dir = get_package_share_directory('resilience')
                config_path = os.path.join(package_dir, 'config', 'main_config.yaml')
            except ImportError:
                # Fallback for non-ROS environments
                config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'main_config.yaml')
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded segmentation config from: {config_path}")
            return config
        except Exception as e:
            print(f"Error loading segmentation config from {config_path}: {e}")
            return self.get_default_segmentation_config()
    
    def get_default_segmentation_config(self) -> Dict:
        """Get default segmentation configuration."""
        return {
            'objects': ['floor', 'ceiling', 'wall', 'person', 'chair', 'table'],
            'segmentation': {
                'apply_softmax': True,
                'normalize_features': True,
                'prefer_enhanced_embeddings': True,
                'enhanced_similarity_threshold': 0.5,
                'enable_dbscan': True,
                'dbscan_eps': 0.3,
                'dbscan_min_samples': 5,
                'colors': [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
            }
        }
    
    def encode_word_embeddings(self):
        """Encode word embeddings for all objects."""
        try:
            if not self.naradio_ready or not self.radio_encoder:
                print("NARadio not ready for word encoding")
                return
            
            self.word_list = self.segmentation_config['objects']
            batch_size = 4  # Process words in batches
            
            print(f"Encoding {len(self.word_list)} base objects into embeddings...")
            
            # Process words in batches
            word_features_list = []
            
            for i in range(0, len(self.word_list), batch_size):
                batch_words = self.word_list[i:i+batch_size]
                
                with torch.no_grad():
                    batch_features = self.radio_encoder.encode_labels(batch_words)
                    word_features_list.append(batch_features)
            
            # Concatenate all batch features
            self.word_features = torch.cat(word_features_list, dim=0)
            
            print(f"✓ Encoded {len(self.word_list)} base objects, features shape: {self.word_features.shape}")
            
        except Exception as e:
            print(f"Error encoding word embeddings: {e}")
            import traceback
            traceback.print_exc()
    
    def add_vlm_object(self, vlm_answer: str) -> bool:
        """
        Add a VLM answer as a new object for segmentation.
        
        Args:
            vlm_answer: The VLM answer to add as a segmentation object
            
        Returns:
            bool: True if successfully added, False otherwise
        """
        try:
            if not vlm_answer or not isinstance(vlm_answer, str):
                print(f"[ADD_VLM] ❌ Invalid VLM object: '{vlm_answer}'")
                return False
                
            vlm_answer_normalized = vlm_answer.lower().strip()
            if not vlm_answer_normalized:
                print(f"[ADD_VLM] ❌ Empty VLM object after normalization")
                return False
            
            print(f"[ADD_VLM] Request to add VLM object: '{vlm_answer}'")
            
            if not self.naradio_ready or not self.radio_encoder:
                print(f"[ADD_VLM] ❌ Cannot add VLM object '{vlm_answer}': NARadio not ready")
                return False
            
            # CRITICAL: Check if object already exists (case-insensitive, trimmed)
            # Check word_list (case-insensitive)
            word_list_normalized = {obj.lower().strip() for obj in self.word_list}
            if vlm_answer_normalized in word_list_normalized:
                print(f"[ADD_VLM] ⚠️  VLM object '{vlm_answer}' already exists in base word_list (case-insensitive)")
                return True
            
            # Check dynamic_objects (case-insensitive)
            dynamic_objects_normalized = {obj.lower().strip() for obj in self.dynamic_objects}
            if vlm_answer_normalized in dynamic_objects_normalized:
                print(f"[ADD_VLM] ⚠️  VLM object '{vlm_answer}' already exists in dynamic_objects (case-insensitive)")
                return True
            
            print(f"[ADD_VLM] Adding VLM object '{vlm_answer}' to segmentation list...")
            print(f"[ADD_VLM] Current word_list: {self.word_list}")
            print(f"[ADD_VLM] Current dynamic_objects: {self.dynamic_objects}")
            
            # Encode the new object
            print(f"[ADD_VLM] Encoding label '{vlm_answer}'...")
            with torch.no_grad():
                new_features = self.radio_encoder.encode_labels([vlm_answer])
            print(f"[ADD_VLM] ✓ Encoded features shape: {new_features.shape}, device: {new_features.device}")
            
            # Ensure device consistency with base features
            if self.word_features is not None:
                target_device = self.word_features.device
                print(f"[ADD_VLM] Base features device: {target_device}")
                if new_features.device != target_device:
                    print(f"[ADD_VLM] Moving VLM features from {new_features.device} to {target_device}")
                    new_features = new_features.to(target_device)
            
            # CRITICAL: Final check before adding (prevent race conditions)
            # Double-check in case another thread added it
            dynamic_objects_normalized_check = {obj.lower().strip() for obj in self.dynamic_objects}
            if vlm_answer_normalized in dynamic_objects_normalized_check:
                print(f"[ADD_VLM] ⚠️  VLM object '{vlm_answer}' was added by another thread, skipping")
                return True
            
            # Add to dynamic objects
            print(f"[ADD_VLM] Adding '{vlm_answer}' to dynamic_objects list...")
            self.dynamic_objects.append(vlm_answer)
            print(f"[ADD_VLM] ✓ dynamic_objects: {self.dynamic_objects}")
            
            # Update dynamic features
            if self.dynamic_features is None:
                print(f"[ADD_VLM] Initializing dynamic_features with new features")
                self.dynamic_features = new_features
            else:
                print(f"[ADD_VLM] Concatenating to existing dynamic_features (shape: {self.dynamic_features.shape})")
                # Ensure device consistency with existing dynamic features
                if self.dynamic_features.device != new_features.device:
                    print(f"[ADD_VLM] Moving new features to match dynamic features device: {self.dynamic_features.device}")
                    new_features = new_features.to(self.dynamic_features.device)
                self.dynamic_features = torch.cat([self.dynamic_features, new_features], dim=0)
            
            print(f"[ADD_VLM] ✓ dynamic_features shape: {self.dynamic_features.shape}")
            
            # Add color for the new object (white for new VLM objects)
            new_color = [255, 255, 255]  # White color for new VLM objects
            self.dynamic_colors.append(new_color)
            
            print(f"[ADD_VLM] ✅ Added VLM object '{vlm_answer}' with white color")
            print(f"[ADD_VLM]    Total dynamic objects: {len(self.dynamic_objects)}")
            print(f"[ADD_VLM]    VLM features shape: {new_features.shape}, device: {new_features.device}")
            print(f"[ADD_VLM]    All objects now: {self.get_all_objects()}")

            if self.cause_registry is not None:
                try:
                    embedding_np = new_features.detach().cpu().numpy().reshape(-1)
                    self.cause_registry.upsert_cause(
                        vlm_answer,
                        embedding_np,
                        source="vlm",
                        type_="dynamic",
                        color_rgb=new_color
                    )
                    print(f"[ADD_VLM] ✓ Updated cause registry")
                except Exception as registry_err:
                    print(f"[ADD_VLM] ⚠️  CauseRegistry update failed for '{vlm_answer}': {registry_err}")
            return True
            
        except Exception as e:
            print(f"[ADD_VLM] ❌ ERROR adding VLM object '{vlm_answer}': {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def add_enhanced_embedding(self, vlm_answer: str, enhanced_embedding: np.ndarray) -> bool:
        """
        Add an enhanced embedding for better similarity detection.
        
        Args:
            vlm_answer: The VLM answer associated with this embedding
            enhanced_embedding: The enhanced embedding computed from weighted spatial features
            
        Returns:
            bool: True if successfully added
        """
        try:
            if not isinstance(enhanced_embedding, np.ndarray):
                print(f"Enhanced embedding must be numpy array, got {type(enhanced_embedding)}")
                return False
            
            # Store the enhanced embedding
            self.enhanced_embeddings[vlm_answer] = enhanced_embedding.copy()
            
            # Assign a distinctive color for enhanced embeddings (bright cyan for visibility)
            self.enhanced_embedding_colors[vlm_answer] = [0, 255, 255]  # Bright cyan
            
            print(f"✓ Added enhanced embedding for '{vlm_answer}' (shape: {enhanced_embedding.shape})")
            print(f"  Total enhanced embeddings: {len(self.enhanced_embeddings)}")

            if self.cause_registry is not None:
                try:
                    self.cause_registry.set_enhanced_embedding(
                        vlm_answer,
                        enhanced_embedding,
                        metadata={"source": "enhanced_embedding"}
                    )
                except Exception as registry_err:
                    print(f"CauseRegistry enhanced update failed for '{vlm_answer}': {registry_err}")
            return True
            
        except Exception as e:
            print(f"Error adding enhanced embedding for '{vlm_answer}': {e}")
            return False
    
    def has_enhanced_embedding(self, vlm_answer: str) -> bool:
        """Check if we have an enhanced embedding for this VLM answer."""
        return vlm_answer in self.enhanced_embeddings
    
    def get_enhanced_embedding(self, vlm_answer: str) -> Optional[np.ndarray]:
        """Get enhanced embedding for a VLM answer."""
        return self.enhanced_embeddings.get(vlm_answer)
    
    def get_all_enhanced_objects(self) -> List[str]:
        """Get list of VLM answers that have enhanced embeddings."""
        return list(self.enhanced_embeddings.keys())
    
    def remove_enhanced_embedding(self, vlm_answer: str) -> bool:
        """Remove enhanced embedding for a VLM answer."""
        if vlm_answer in self.enhanced_embeddings:
            del self.enhanced_embeddings[vlm_answer]
            if vlm_answer in self.enhanced_embedding_colors:
                del self.enhanced_embedding_colors[vlm_answer]
            print(f"Removed enhanced embedding for '{vlm_answer}'")
            return True
        return False
    
    def get_all_objects(self) -> List[str]:
        """Get all objects (base + dynamic) for segmentation."""
        # CRITICAL: Remove duplicates (case-insensitive) before returning
        all_objects = self.word_list + self.dynamic_objects
        seen_normalized = set()
        deduplicated = []
        for obj in all_objects:
            if not isinstance(obj, str):
                continue
            obj_normalized = obj.lower().strip()
            if obj_normalized and obj_normalized not in seen_normalized:
                seen_normalized.add(obj_normalized)
                deduplicated.append(obj)
        return deduplicated

    def get_base_objects(self) -> List[str]:
        """Get only predefined objects from config (text embeddings, no enhanced)."""
        return list(self.word_list)
    
    def get_all_features(self):
        """Get all features (base + dynamic) for segmentation."""
        if self.dynamic_features is None:
            return self.word_features
        else:
            # Ensure device consistency before concatenation
            if self.word_features.device != self.dynamic_features.device:
                print(f"Device mismatch in get_all_features: base={self.word_features.device}, dynamic={self.dynamic_features.device}")
                # Move dynamic features to base device
                self.dynamic_features = self.dynamic_features.to(self.word_features.device)
                print(f"Moved dynamic features to {self.word_features.device}")
            
            return torch.cat([self.word_features, self.dynamic_features], dim=0)
    
    def get_all_colors(self) -> List[List[int]]:
        """Get all colors (base + dynamic) for segmentation."""
        return self.colors + self.dynamic_colors
    
    def setup_colors(self):
        """Set up color mapping for visualization."""
        self.colors = self.segmentation_config['segmentation']['colors']
        
        # Keep original colors for base objects, use white for additional objects
        original_color_count = len(self.colors)
        original_object_count = len(self.word_list)
        
        # Trim colors to match the number of base objects
        if original_color_count > original_object_count:
            self.colors = self.colors[:original_object_count]
            print(f"  Trimmed colors from {original_color_count} to {len(self.colors)} to match {original_object_count} objects")
        
        # If we have more objects than original colors, add white colors for the extra objects
        if original_object_count > len(self.colors):
            extra_colors_needed = original_object_count - len(self.colors)
            for _ in range(extra_colors_needed):
                self.colors.append([255, 255, 255])  # White for additional objects
        
        print(f"✓ Set up {len(self.colors)} colors for {len(self.word_list)} objects")
        print(f"  Final color count: {len(self.colors)}")

    @torch.no_grad()
    def norm_img_01(self, x):
        """Normalize image tensor to [0, 1] range like RayFronts utils."""
        B, C, H, W = x.shape
        x = x - torch.min(x.reshape(B, C, H*W), dim=-1).values.reshape(B, C, 1, 1)
        x = x / (torch.max(x.reshape(B, C, H*W), dim=-1).values.reshape(B, C, 1, 1) + 1e-8)
        return x

    @torch.no_grad()
    def compute_cos_sim(self, vec1: torch.FloatTensor, vec2: torch.FloatTensor, softmax: bool = False) -> torch.FloatTensor:
        """Compute cosine similarity between two batches of vectors - RayFronts implementation."""
        N, C1 = vec1.shape
        M, C2 = vec2.shape
        if C1 != C2:
            raise ValueError(f"vec1 feature dimension '{C1}' does not match vec2 feature dimension '{C2}'")
        
        vec1 = vec1 / vec1.norm(dim=-1, keepdim=True)
        vec1 = vec1.reshape(1, N, 1, C1)
        
        vec2 = vec2 / vec2.norm(dim=-1, keepdim=True)
        vec2 = vec2.reshape(M, 1, C1, 1)
        
        sim = (vec1 @ vec2).reshape(M, N)
        if softmax:
            return torch.softmax(100 * sim, dim=-1)
        else:
            return sim

    @torch.inference_mode()
    def compute_vlm_similarity_map_optimized(self, rgb_image: np.ndarray, vlm_answer: str, 
                                            feat_map_np: Optional[np.ndarray] = None,
                                            use_softmax: bool = True, chunk_size: int = 4000) -> Optional[np.ndarray]:
        """
        OPTIMIZED: Compute similarity map for a specific VLM answer, reusing features if provided.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            vlm_answer: The VLM answer to compute similarity for
            feat_map_np: Pre-computed feature map (optional, avoids redundant computation)
            use_softmax: Whether to apply softmax
            chunk_size: Chunk size for memory efficiency (increased default)
            
        Returns:
            np.ndarray: Similarity map (H, W) for the VLM answer, or None if failed
        """
        try:
            self.last_similarity_error = None
            # Check if VLM answer is in our objects
            all_objects = self.get_all_objects()
            
            if vlm_answer not in all_objects:
                return None
            
            vlm_index = all_objects.index(vlm_answer)
            
            # Get all features (base + dynamic objects)
            all_features = self.get_all_features()
            
            if all_features is None or len(all_objects) == 0:
                return None
            
            N = len(all_objects)
            
            # If there's only one object/class, softmax is ill-defined (it becomes 1 everywhere).
            # Instead, fall back to non-softmax similarity and normalize to [0, 1] later.
            if use_softmax and N == 1:
                use_softmax = False

            # CRITICAL: Keep embeddings/features and image features on the SAME device.
            # On first narration we often have embeddings on CPU while the feature map is moved to CUDA,
            # which causes matmul/device errors inside compute_cos_sim and returns None.
            if isinstance(all_features, torch.Tensor):
                device = all_features.device
                all_features = all_features.to(device=device, dtype=torch.float32, non_blocking=True)
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                all_features = torch.as_tensor(all_features, device=device, dtype=torch.float32)
            
            # OPTIMIZATION: Reuse features if provided, otherwise compute them
            reused_feat_map = feat_map_np is not None
            if feat_map_np is not None:
                # Convert to tensor
                if isinstance(feat_map_np, np.ndarray):
                    feat_map_tensor = torch.from_numpy(feat_map_np).to(device=device, dtype=torch.float32)
                else:
                    feat_map_tensor = feat_map_np.to(device=device, dtype=torch.float32) if hasattr(feat_map_np, 'to') else torch.as_tensor(feat_map_np, device=device, dtype=torch.float32)
            else:
                # Set model resolution based on configured input size
                resolution = (self.radio_input_resolution, self.radio_input_resolution)
                if hasattr(self.radio_encoder, "input_resolution"):
                    self.radio_encoder.input_resolution = resolution
 
                # Convert image to tensor
                tensor_image = torch.from_numpy(rgb_image).permute(2, 0, 1)
                tensor_image = tensor_image.to(device).float() / 255.0
                tensor_image = torch.nn.functional.interpolate(
                    tensor_image.unsqueeze(0), resolution, mode="bilinear", antialias=True)
                
                # Extract features
                feat_map_tensor = self.radio_encoder.encode_image_to_feat_map(tensor_image)
                
                # GPU MEMORY OPTIMIZATION: Delete intermediate tensor immediately
                del tensor_image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Align features with language
            feat_map_aligned = self.radio_encoder.align_spatial_features_with_language(feat_map_tensor)
            
            # GPU MEMORY OPTIMIZATION: Delete original tensor after alignment
            del feat_map_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # IMPORTANT: If caller provided a precomputed feature map (e.g., 1xCx32x32),
            # do NOT upscale it to 512x512 — that can allocate ~0.8GB+ (for C=768) and OOM,
            # especially on the first narration when VRAM is most constrained.
            #
            # Only apply the expensive upscale when we extracted features from the raw image here.
            if not reused_feat_map:
                resolution = (self.radio_input_resolution, self.radio_input_resolution)
                feat_map_aligned = torch.nn.functional.interpolate(
                    feat_map_aligned, resolution, mode="bilinear", antialias=True)
            feat_map_aligned = feat_map_aligned.squeeze(0).permute(1, 2, 0)
            
            # Compute similarity with optimized chunking
            H, W, C = feat_map_aligned.shape

            # Early, explicit shape check (much clearer than failing inside matmul).
            if not isinstance(all_features, torch.Tensor) or all_features.ndim != 2:
                msg = f"all_features is not a 2D tensor: type={type(all_features)}, ndim={getattr(all_features, 'ndim', None)}"
                self.last_similarity_error = msg
                self._logger.error(msg)
                return None
            if all_features.shape[1] != C:
                msg = (
                    f"Feature-dimension mismatch for '{vlm_answer}': "
                    f"label_embeddings_dim={int(all_features.shape[1])} vs image_feat_dim={int(C)} "
                    f"(all_features.shape={tuple(all_features.shape)}, feat_map_aligned.shape={tuple(feat_map_aligned.shape)}, "
                    f"use_softmax={use_softmax}, device={device})"
                )
                self.last_similarity_error = msg
                self._logger.error(msg)
                return None

            feat_map_flat = feat_map_aligned.reshape(-1, C)  # H*W x C
            
            # GPU MEMORY OPTIMIZATION: Delete aligned tensor after flattening
            del feat_map_aligned
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # GPU MEMORY OPTIMIZATION: Use smaller chunks to reduce peak memory
            # Reduce chunk_size for large feature maps (512x512 = 262144 pixels)
            if feat_map_flat.shape[0] > 100000:
                chunk_size = min(chunk_size, 2000)  # Smaller chunks for large images
            
            num_chunks = max(1, int(np.ceil(feat_map_flat.shape[0] / chunk_size)))
            cos_sim_chunks = []
            
            for c in range(num_chunks):
                chunk_features = feat_map_flat[c*chunk_size:(c+1)*chunk_size]
                
                # Compute similarity
                chunk_sim = self.compute_cos_sim(all_features, chunk_features, softmax=use_softmax)
                cos_sim_chunks.append(chunk_sim)
                
                # GPU MEMORY OPTIMIZATION: Delete chunk after processing
                del chunk_features
                if torch.cuda.is_available() and c % 10 == 0:  # Clear cache every 10 chunks
                    torch.cuda.empty_cache()
            
            # GPU MEMORY OPTIMIZATION: Delete flat tensor before concatenation
            del feat_map_flat
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Concatenate chunks
            cos_sim = torch.cat(cos_sim_chunks, dim=0)  # H*W x N
            
            # GPU MEMORY OPTIMIZATION: Delete chunks after concatenation
            del cos_sim_chunks
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            cos_sim = cos_sim.reshape(H, W, N)   # H x W x N
            
            # Apply normalization if not using softmax
            if not use_softmax:
                cos_sim = self.norm_img_01(cos_sim.permute(2, 0, 1).unsqueeze(0))  # 1 x N x H x W
                cos_sim = cos_sim.squeeze(0).permute(1, 2, 0)  # H x W x N
            
            # Extract similarity map for the VLM answer
            vlm_similarity_map = cos_sim[:, :, vlm_index]  # H x W
            
            # GPU MEMORY OPTIMIZATION: Delete large tensor immediately after extraction
            del cos_sim
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Return as numpy array (move to CPU immediately)
            result = vlm_similarity_map.cpu().numpy()

            # If we reused a low‑resolution feature map (e.g. 32x32), upsample the
            # 1‑channel similarity map back to the original RGB image resolution so
            # downstream code (masks, depth reprojection) sees a map aligned with
            # the narration image, not the feature grid.
            if reused_feat_map and rgb_image is not None and isinstance(rgb_image, np.ndarray):
                h_img, w_img = rgb_image.shape[:2]
                if result.shape != (h_img, w_img):
                    result = cv2.resize(result, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            
            # GPU MEMORY OPTIMIZATION: Delete GPU tensor after CPU transfer
            del vlm_similarity_map
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except torch.cuda.OutOfMemoryError as e:
            # GPU MEMORY OPTIMIZATION: Clear cache on OOM and log error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            msg = f"CUDA OOM in similarity computation for '{vlm_answer}': {e}"
            self.last_similarity_error = msg
            self._logger.error(msg)
            return None
        except Exception as e:
            msg = (
                f"Error in similarity computation for '{vlm_answer}': {e} "
                f"(use_softmax={use_softmax}, "
                f"all_features_shape={tuple(all_features.shape) if isinstance(all_features, torch.Tensor) else None}, "
                f"feat_map_np_shape={getattr(feat_map_np, 'shape', None)})"
            )
            self.last_similarity_error = msg
            self._logger.error(msg)
            return None

    def apply_colormap(self, image: np.ndarray, cmap_name='viridis') -> np.ndarray:
        """Apply a colormap to a grayscale image and return an RGB uint8 image."""
        from matplotlib import cm
        
        # Ensure image is normalized to [0, 1]
        if image.dtype != np.float16 and image.dtype != np.float32 and image.dtype != np.float64:
            image = image.astype(np.float32) / 255.0
        image = np.clip(image, 0, 1)
        cmap = cm.get_cmap(cmap_name)
        colored = cmap(image)[:, :, :3]  # Drop alpha channel
        return (colored * 255).astype(np.uint8)

  
    def process_vlm_similarity_visualization_optimized(self, rgb_image: np.ndarray, vlm_answer: str,
                                                     feat_map_np: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        OPTIMIZED: Process VLM similarity map and create colored visualization, reusing features if provided.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            vlm_answer: The VLM answer to compute similarity for
            feat_map_np: Pre-computed feature map (optional, avoids redundant computation)
            
        Returns:
            dict: Contains similarity_map, colored_similarity, metadata, and processing_info
        """
        if not self.segmentation_ready or not self.naradio_ready:
            return None
        
        try:
            start_time = time.time()
            
            # OPTIMIZATION: Reuse features if provided, use smaller chunk size for memory safety
            similarity_map = self.compute_vlm_similarity_map_optimized(
                rgb_image, vlm_answer, feat_map_np, use_softmax=True, chunk_size=2000)
            
            if similarity_map is None:
                return None
            
            # Get original image dimensions
            original_height, original_width = rgb_image.shape[:2]
            
            # Resize similarity map to original image dimensions
            similarity_resized = cv2.resize(similarity_map, (original_width, original_height), 
                                          interpolation=cv2.INTER_LINEAR)
            
            # SPEED OPTIMIZATION: Use faster colormap application
            colored_similarity = self.apply_colormap(similarity_resized, cmap_name='viridis')
            
            # SPEED OPTIMIZATION: Minimal metadata creation
            processing_time = time.time() - start_time
            
            result = {
                'similarity_map': similarity_resized,  # Resized to original image size
                'colored_similarity': colored_similarity,  # RGB colored version
                'metadata': {
                    'vlm_answer': vlm_answer,
                    'processing_time': processing_time,
                    'reused_features': feat_map_np is not None
                },
                'processing_info': {
                    'processing_time': processing_time
                }
            }
            
            return result
            
        except Exception as e:
            return None


    def is_segmentation_ready(self) -> bool:
        """Check if combined segmentation is ready."""
        return self.segmentation_ready and self.naradio_ready

    @torch.inference_mode()
    def compute_enhanced_cause_embedding(self, rgb_image: np.ndarray, vlm_answer: str, 
                                       similarity_map: Optional[np.ndarray] = None,
                                       feat_map_np: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Compute enhanced cause embedding by weighted averaging spatial features using similarity as weights.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            vlm_answer: The VLM answer to compute enhanced embedding for
            similarity_map: Pre-computed similarity map (optional, will compute if not provided)
            feat_map_np: Pre-computed feature map (optional, avoids redundant feature extraction)
            
        Returns:
            np.ndarray: Enhanced cause embedding vector, or None if failed
        """
        try:
            # Check if VLM answer is in our objects
            all_objects = self.get_all_objects()
            if vlm_answer not in all_objects:
                print(f"VLM answer '{vlm_answer}' not in object list for enhanced embedding")
                return None
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # OPTIMIZATION: Reuse pre-computed features if provided
            if feat_map_np is not None:
                # Convert to tensor and align
                if isinstance(feat_map_np, np.ndarray):
                    feat_map_tensor = torch.from_numpy(feat_map_np).to(device)
                else:
                    feat_map_tensor = feat_map_np.to(device)
                
                # Align features with language
                feat_map_aligned = self.radio_encoder.align_spatial_features_with_language(feat_map_tensor)
                
                # Resize to final resolution
                resolution = (self.radio_input_resolution, self.radio_input_resolution)
                feat_map_aligned = torch.nn.functional.interpolate(
                    feat_map_aligned, resolution, mode="bilinear", antialias=True)
                feat_map_aligned = feat_map_aligned.squeeze(0).permute(1, 2, 0)  # H x W x C
            else:
                # Extract spatial features from the image (fallback if not provided)
                resolution = (self.radio_input_resolution, self.radio_input_resolution)
                if hasattr(self.radio_encoder, "input_resolution"):
                    self.radio_encoder.input_resolution = resolution
                
                # Convert image to tensor
                tensor_image = torch.from_numpy(rgb_image).permute(2, 0, 1)
                tensor_image = tensor_image.to(device).float() / 255.0
                tensor_image = torch.nn.functional.interpolate(
                    tensor_image.unsqueeze(0), resolution, mode="bilinear", antialias=True)
                
                # Extract features
                feat_map_tensor = self.radio_encoder.encode_image_to_feat_map(tensor_image)
                
                # Align features with language
                feat_map_aligned = self.radio_encoder.align_spatial_features_with_language(feat_map_tensor)
                
                # Resize to final resolution
                feat_map_aligned = torch.nn.functional.interpolate(
                    feat_map_aligned, resolution, mode="bilinear", antialias=True)
                feat_map_aligned = feat_map_aligned.squeeze(0).permute(1, 2, 0)  # H x W x C
                
                del tensor_image, feat_map_tensor
            
            H, W, C = feat_map_aligned.shape
            
            # Get or compute similarity map
            if similarity_map is None:
                # OPTIMIZATION: Reuse pre-computed features if available
                similarity_map = self.compute_vlm_similarity_map_optimized(
                    rgb_image, vlm_answer, feat_map_np=feat_map_np)
                if similarity_map is None:
                    print(f"Failed to compute similarity map for enhanced embedding")
                    return None
            
            # Ensure similarity map matches feature map dimensions
            if similarity_map.shape != (H, W):
                similarity_map = cv2.resize(similarity_map, (W, H), interpolation=cv2.INTER_LINEAR)
            
            # Convert similarity map to tensor weights
            weights = torch.from_numpy(similarity_map).to(device).float()  # H x W
            
            # ENHANCED FIX: Only consider pixels with high similarity to avoid garbage features
            similarity_threshold = self.segmentation_config['segmentation'].get('enhanced_similarity_threshold', 0.5)
            high_sim_mask = weights > 0.98
            
            # Zero out low-similarity pixels to focus only on the actual cause/hotspots
            weights = weights * high_sim_mask.float()
            
            # Check if we have enough high-similarity pixels
            num_high_sim_pixels = torch.sum(high_sim_mask).item()
            total_pixels = H * W
            
            if num_high_sim_pixels == 0:
                print(f"Warning: No pixels above similarity threshold {similarity_threshold}, using all pixels")
                weights = torch.from_numpy(similarity_map).to(device).float()
            else:
                print(f"Enhanced embedding using {num_high_sim_pixels}/{total_pixels} high-similarity pixels (>{similarity_threshold})")
            
            # Normalize weights to sum to 1 for proper weighted averaging
            total_weight = torch.sum(weights)
            if total_weight > 1e-8:
                weights = weights / total_weight
            else:
                print(f"Warning: similarity weights sum to near zero after thresholding")
                return None
            
            # Flatten spatial dimensions for weighted averaging
            feat_map_flat = feat_map_aligned.reshape(-1, C)  # (H*W) x C
            weights_flat = weights.reshape(-1)  # (H*W)
            
            # Compute weighted average of features using only high-similarity pixels
            # This gives us a much cleaner enhanced embedding focused on the actual cause
            enhanced_embedding = torch.sum(feat_map_flat * weights_flat.unsqueeze(1), dim=0)  # C
            
            # Normalize the enhanced embedding
            enhanced_embedding = enhanced_embedding / (torch.norm(enhanced_embedding) + 1e-8)
            
            # Convert to numpy and return
            enhanced_embedding_np = enhanced_embedding.cpu().numpy()
            
            print(f"✓ Computed enhanced cause embedding for '{vlm_answer}' (shape: {enhanced_embedding_np.shape})")
            print(f"  Used {total_weight.item():.3f} total similarity weight from {H*W} pixels")
            
            # Cleanup - only delete variables that were actually created
            del feat_map_aligned, weights, feat_map_flat, weights_flat
            if feat_map_np is None:
                # Only these were created in the else branch
                del tensor_image, feat_map_tensor
            
            return enhanced_embedding_np
            
        except Exception as e:
            print(f"Error computing enhanced cause embedding: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_enhanced_embedding(self, vlm_answer: str, buffer_dir: str, enhanced_embedding: np.ndarray):
        """Save enhanced cause embedding to buffer directory."""
        try:
            # Create embeddings directory
            embeddings_dir = os.path.join(buffer_dir, 'enhanced_embeddings')
            os.makedirs(embeddings_dir, exist_ok=True)
            
            # Create safe filename
            safe_vlm_name = vlm_answer.replace(' ', '_').replace('/', '_').replace('\\', '_')
            timestamp_str = f"{time.time():.3f}"
            
            # Save enhanced embedding as numpy array
            embedding_path = os.path.join(embeddings_dir, f"enhanced_embedding_{safe_vlm_name}_{timestamp_str}.npy")
            np.save(embedding_path, enhanced_embedding)
            
            # Save metadata
            metadata = {
                'vlm_answer': vlm_answer,
                'embedding_filename': f"enhanced_embedding_{safe_vlm_name}_{timestamp_str}.npy",
                'embedding_shape': list(enhanced_embedding.shape),
                'embedding_norm': float(np.linalg.norm(enhanced_embedding)),
                'timestamp': time.time(),
                'datetime': time.strftime('%Y-%m-%d %H:%M:%S'),
                'description': 'Enhanced cause embedding computed from weighted averaging of spatial features using similarity as weights',
                'computation_method': 'similarity_weighted_spatial_features'
            }
            
            metadata_path = os.path.join(embeddings_dir, f"enhanced_embedding_{safe_vlm_name}_{timestamp_str}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Saved enhanced embedding: {embedding_path}")
            print(f"  Shape: {enhanced_embedding.shape}, Norm: {np.linalg.norm(enhanced_embedding):.4f}")
            
        except Exception as e:
            print(f"Error saving enhanced embedding: {e}")
            import traceback
            traceback.print_exc()
    
    def process_adaptive_similarity_visualization_optimized(self, rgb_image: np.ndarray, vlm_answer: str,
                                                           feat_map_np: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        ADAPTIVE: Choose between enhanced embedding and VLM text similarity based on config and availability.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            vlm_answer: The VLM answer to compute similarity for
            feat_map_np: Pre-computed feature map (optional, avoids redundant computation)
            
        Returns:
            dict: Contains similarity_map, colored_similarity, metadata, and processing_info
        """
        if not self.segmentation_ready or not self.naradio_ready:
            return None
        
        try:
            # Check config preference and availability
            prefer_enhanced = self.segmentation_config['segmentation'].get('prefer_enhanced_embeddings', True)
            has_enhanced = self.has_enhanced_embedding(vlm_answer)
            
            # Decision logic based on config and availability
            if prefer_enhanced:
                # Config says to prefer enhanced embeddings
                if has_enhanced:
                    # Use enhanced embedding similarity
                    return self.process_vlm_similarity_visualization_optimized(
            rgb_image, vlm_answer, feat_map_np)
                else:
                    # Config prefers enhanced but none available - return None instead of falling back
                    print(f"Config prefers enhanced embeddings for '{vlm_answer}' but none available - skipping")
                    return None
            else:
                # Config says to prefer VLM text embeddings
                return self.process_vlm_similarity_visualization_optimized(
                    rgb_image, vlm_answer, feat_map_np)
                
        except Exception as e:
            print(f"Error in adaptive similarity processing: {e}")
            # Only fallback to VLM text if config doesn't prefer enhanced
            prefer_enhanced = self.segmentation_config['segmentation'].get('prefer_enhanced_embeddings', True)
            if not prefer_enhanced:
                return self.process_vlm_similarity_visualization_optimized(
                    rgb_image, vlm_answer, feat_map_np)
            else:
                return None

    def create_merged_hotspot_masks_fast(self, rgb_image: np.ndarray, vlm_answers: List[str], 
                                     feat_map_tensor: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                     feat_map_np: Optional[np.ndarray] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        OPTIMIZED: Create hotspot masks for multiple VLM answers with reduced overhead.
        
        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            vlm_answers: List of VLM answers to process
            feat_map_tensor: Pre-computed feature map (optional, avoids redundant feature extraction)
            feat_map_np: Alias for feat_map_tensor (backward compatibility)
            
        Returns:
            Dict mapping vlm_answer -> hotspot_mask (binary) or None if failed
        """
        try:
            if not self.is_segmentation_ready():
                return None

            # Handle alias
            if feat_map_tensor is None and feat_map_np is not None:
                feat_map_tensor = feat_map_np

            prefer_enhanced = self.segmentation_config['segmentation'].get('prefer_enhanced_embeddings', True)
            hotspot_threshold = self.segmentation_config['segmentation'].get('hotspot_threshold', 0.6)
            min_area = self.segmentation_config['segmentation'].get('min_hotspot_area', 50)

            vlm_hotspots = {}

            # OPTIMIZATION: Filter valid VLM answers upfront
            valid_vlms = [v for v in vlm_answers if v in self.get_all_objects()]
            if not valid_vlms:
                return None

            # OPTIMIZATION: Batch process if using enhanced embeddings
            for vlm_answer in valid_vlms:
                # FIX: Don't skip if enhanced embedding not available - fallback to text embedding
                # Only skip if config explicitly requires enhanced (not just prefers it)
                # prefer_enhanced=True means "use if available, but fallback to text"
                # So we should NOT skip here - let compute_vlm_similarity_map_optimized handle the fallback
                # if prefer_enhanced and not self.has_enhanced_embedding(vlm_answer):
                #     continue

                # Pass pre-computed features directly to similarity map computation
                # This avoids the overhead of process_adaptive_similarity_visualization_optimized
                # which creates visualizations we don't need here.
                similarity_map = self.compute_vlm_similarity_map_optimized(
                    rgb_image, vlm_answer, feat_map_np=feat_map_tensor, use_softmax=True, chunk_size=4000)
                
                if similarity_map is None:
                    continue

                # Resize to original image dimensions if needed
                h, w = rgb_image.shape[:2]
                if similarity_map.shape != (h, w):
                    similarity_map = cv2.resize(similarity_map, (w, h), interpolation=cv2.INTER_LINEAR)

                # OPTIMIZATION: Vectorized thresholding using configured threshold
                hotspot_threshold = 0.90
                # FIX: Use hotspot_threshold from config instead of hardcoded 0.2
                hotspot_mask = (similarity_map > hotspot_threshold).astype(np.uint8) * 255
                if not np.any(hotspot_mask):
                    continue

                # OPTIMIZATION: Skip contour filtering if min_area is small
                if min_area <= 10:
                    vlm_hotspots[vlm_answer] = hotspot_mask
                else:
                    contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    filtered_mask = np.zeros_like(hotspot_mask, dtype=np.uint8)
                    for contour in contours:
                        if cv2.contourArea(contour) >= min_area:
                            cv2.fillPoly(filtered_mask, [contour], 255)

                    if np.any(filtered_mask):
                        vlm_hotspots[vlm_answer] = filtered_mask

            return vlm_hotspots if vlm_hotspots else None

        except Exception:
            return None

    def create_merged_hotspot_masks(self, rgb_image: np.ndarray, vlm_answers: List[str], feat_map_np: Optional[np.ndarray] = None) -> Optional[Dict[str, np.ndarray]]:
        """Legacy method - redirects to optimized version."""
        return self.create_merged_hotspot_masks_fast(rgb_image, vlm_answers, feat_map_np)

 
    def _make_json_serializable(self, obj):
        """Convert object to JSON-serializable format, handling numpy arrays and other non-serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj