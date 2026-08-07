import numpy as np
import cv2

def constrained_watershed(image, prob_maps, gate_mask, tau_core=0.5, tau_bound=0.5, kernel_bridge_size=(9, 9)):
    """
    Applies cv2.watershed constrained by predicted Core/Boundary probabilities and the Spatial Gate B(x).
    
    Args:
        image: Original RGB image as numpy array [H, W, 3] (uint8)
        prob_maps: Model probabilities [3, H, W] for (BG=0, Boundary=1, Core=2)
        gate_mask: Spatial Gate B(x) mask [H, W] (1 inside boxes, 0 outside)
        tau_core: Threshold for Core seeds
        tau_bound: Threshold for Boundary barriers
        kernel_bridge_size: Kernel dimensions for morphological seed bridging (default: (9, 9))
        
    Returns:
        instance_mask: Segmentation mask with unique integer IDs for each instance [H, W]
    """
    if len(prob_maps.shape) == 3:
        # Expected [3, H, W]
        p_bg = prob_maps[0]
        p_bound = prob_maps[1]
        p_core = prob_maps[2]
    else:
        raise ValueError("prob_maps should be shape [3, H, W]")
        
    H, W = p_core.shape
    
    # 1. Generate Seeds Map S(x)
    # S(x) = (P(Core) > tau_core) AND (Gate == 1)
    core_binary = np.logical_and(p_core > tau_core, gate_mask == 1).astype(np.uint8)
    
    # FIX 2: Morphological Seed Bridging / Closing Operation
    # Connects adjacent fragmented P(Core) seeds on the same silkworm into a single continuous seed
    if kernel_bridge_size is not None and kernel_bridge_size[0] > 0 and kernel_bridge_size[1] > 0:
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_bridge_size)
        core_binary = cv2.morphologyEx(core_binary, cv2.MORPH_CLOSE, kernel_bridge)
    
    # Find connected components to give each seed a unique ID
    num_labels, markers = cv2.connectedComponents(core_binary)
    
    # OpenCV watershed requires seeds to be positive integers.
    # Background should be 0. Unknown region should be 0. 
    # But wait, OpenCV watershed uses 0 for unknown.
    # Let's shift markers so that our background (outside boxes) is safely handled.
    
    # 2. Generate Barrier Map W(x)
    # W(x) = (P(Boundary) > tau_bound) OR (Gate == 0)
    # We want watershed to stop at barriers. 
    # In cv2.watershed, boundaries are ultimately marked as -1. 
    # If we want regions to be strictly background, we can seed them with a specific background ID.
    
    # Let's assign background a specific marker, e.g., max_label + 1
    bg_label = num_labels + 1
    
    # Create the marker image for watershed
    # Initialize all as unknown (0)
    watershed_markers = np.zeros((H, W), dtype=np.int32)
    
    # 1. Assign seeds (1, 2, ... num_labels) to Core regions
    watershed_markers[markers > 0] = markers[markers > 0]
    
    # 2. Assign Background label ONLY outside BBoxes (Gate == 0)
    watershed_markers[gate_mask == 0] = bg_label
    
    # 3. Assign Background label to predicted organic Boundaries (rigid walls)
    predicted_boundaries = (p_bound > tau_bound)
    watershed_markers[predicted_boundaries] = bg_label
    
    # Apply Watershed
    markers_out = cv2.watershed(image.astype(np.uint8), watershed_markers.copy())
    
    # Post-process:
    # Everything marked as bg_label or -1 (watershed boundary lines) becomes 0 (background)
    instance_mask = markers_out.copy()
    instance_mask[instance_mask == bg_label] = 0
    instance_mask[instance_mask == -1] = 0
    
    # The remaining positive integers are the silkworm instances
    return instance_mask
