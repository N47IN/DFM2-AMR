#!/usr/bin/env python3
"""
Simple NARadio Similarity Testing Script

Usage:
    python test_similarity.py --image path/to/image.jpg --caption "object description"
    
Example:
    python test_similarity.py --image test.jpg --caption "box fan"
"""

import sys
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add resilience to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resilience.naradio_processor import NARadioProcessor


def test_similarity(image_path: str, caption: str, threshold: float = 0.6, save_output: bool = True):
    """
    Test similarity map generation for an image and caption.
    
    Args:
        image_path: Path to input image
        caption: Text caption/prompt to compute similarity for
        threshold: Similarity threshold for hotspot detection
        save_output: Whether to save output images
    """
    print("=" * 80)
    print("NARadio Similarity Testing")
    print("=" * 80)
    print(f"Image: {image_path}")
    print(f"Caption: '{caption}'")
    print(f"Threshold: {threshold}")
    print()
    
    # 1. Load image
    print("📷 Loading image...")
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Failed to load image from {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✓ Image loaded (shape: {image_rgb.shape})")
    print()
    
    # 2. Initialize NARadioProcessor
    print("🔧 Initializing NARadioProcessor...")
    processor = NARadioProcessor(
        radio_model_version='radio_v2.5-b',
        radio_lang_model='siglip',
        radio_input_resolution=512,
        enable_visualization=True,
        enable_combined_segmentation=True,
        segmentation_config_path=None  # Use defaults
    )
    
    if not processor.is_ready():
        print("❌ Error: NARadio processor not ready")
        return
    
    if not processor.is_segmentation_ready():
        print("❌ Error: Segmentation not ready")
        return
    
    print("✓ NARadioProcessor initialized")
    print()
    
    # 3. Add caption as VLM object
    print(f"📝 Adding caption '{caption}' as VLM object...")
    success = processor.add_vlm_object(caption)
    if not success:
        print(f"❌ Error: Failed to add VLM object '{caption}'")
        return
    print(f"✓ VLM object added")
    print()
    
    # 4. Extract features from image
    print("🔬 Extracting features from image...")
    feat_map_np, naradio_vis = processor.process_features_optimized(
        image_rgb, 
        need_visualization=True,
        reuse_features=False,
        return_tensor=False
    )
    
    if feat_map_np is None:
        print("❌ Error: Failed to extract features")
        return
    
    print(f"✓ Features extracted (shape: {feat_map_np.shape})")
    print()
    
    # 5. Compute similarity map
    print(f"🎯 Computing similarity map for '{caption}'...")
    similarity_map = processor.compute_vlm_similarity_map_optimized(
        image_rgb,
        caption,
        feat_map_np=feat_map_np,
        use_softmax=True,
        chunk_size=4000
    )
    
    if similarity_map is None:
        print("❌ Error: Failed to compute similarity map")
        return
    
    print(f"✓ Similarity map computed (shape: {similarity_map.shape})")
    print(f"  Min: {np.min(similarity_map):.4f}")
    print(f"  Max: {np.max(similarity_map):.4f}")
    print(f"  Mean: {np.mean(similarity_map):.4f}")
    print()
    
    # 6. Create hotspot mask
    print(f"🎭 Creating hotspot mask (threshold: {threshold})...")
    
    # Resize similarity map to match image dimensions
    h, w = image_rgb.shape[:2]
    similarity_resized = cv2.resize(similarity_map, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Create binary hotspot mask
    hotspot_mask = (similarity_resized > threshold).astype(np.uint8) * 255
    hotspot_pixels = int(np.sum(hotspot_mask > 0))
    
    print(f"✓ Hotspot mask created")
    print(f"  Hotspot pixels: {hotspot_pixels} ({100*hotspot_pixels/(h*w):.2f}%)")
    print()
    
    # 7. Create visualizations
    print("🎨 Creating visualizations...")
    
    # Apply colormap to similarity map
    similarity_colored = processor.apply_colormap(similarity_resized, cmap_name='viridis')
    
    # Create overlay (image + hotspot mask)
    overlay = image_rgb.copy()
    overlay[hotspot_mask > 0] = overlay[hotspot_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
    overlay = overlay.astype(np.uint8)
    
    print("✓ Visualizations created")
    print()
    
    # 8. Display results
    print("📊 Displaying results...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Similarity Testing: "{caption}"', fontsize=16, fontweight='bold')
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # NARadio features visualization
    if naradio_vis is not None:
        axes[0, 1].imshow(naradio_vis)
        axes[0, 1].set_title('NARadio Features (PCA)')
    else:
        axes[0, 1].text(0.5, 0.5, 'No visualization', ha='center', va='center')
        axes[0, 1].set_title('NARadio Features')
    axes[0, 1].axis('off')
    
    # Similarity map (raw)
    im = axes[0, 2].imshow(similarity_resized, cmap='viridis', vmin=0, vmax=1)
    axes[0, 2].set_title('Similarity Map (Raw)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
    
    # Similarity map (colored)
    axes[1, 0].imshow(similarity_colored)
    axes[1, 0].set_title('Similarity Map (Colored)')
    axes[1, 0].axis('off')
    
    # Hotspot mask
    axes[1, 1].imshow(hotspot_mask, cmap='gray')
    axes[1, 1].set_title(f'Hotspot Mask (threshold={threshold})')
    axes[1, 1].axis('off')
    
    # Overlay
    axes[1, 2].imshow(overlay)
    axes[1, 2].set_title(f'Overlay ({hotspot_pixels} pixels)')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # 9. Save outputs
    if save_output:
        output_dir = os.path.join(os.path.dirname(image_path), 'similarity_test_output')
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        safe_caption = caption.replace(' ', '_').replace('/', '_')
        
        # Save figure
        fig_path = os.path.join(output_dir, f"{base_name}_{safe_caption}_results.png")
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved results to: {fig_path}")
        
        # Save individual outputs
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_{safe_caption}_similarity.png"), 
                   (similarity_resized * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_{safe_caption}_hotspot.png"), 
                   hotspot_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_{safe_caption}_overlay.png"), 
                   cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        print(f"💾 Saved individual outputs to: {output_dir}")
        print()
    
    plt.show()
    
    print("=" * 80)
    print("✅ Testing completed successfully!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Test NARadio similarity map generation')
    parser.add_argument('--image', '-i', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--caption', '-c', type=str, required=True,
                       help='Caption/prompt to test similarity for')
    parser.add_argument('--threshold', '-t', type=float, default=0.6,
                       help='Similarity threshold for hotspot detection (default: 0.6)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output images')
    
    args = parser.parse_args()
    
    try:
        test_similarity(
            image_path=args.image,
            caption=args.caption,
            threshold=args.threshold,
            save_output=not args.no_save
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
