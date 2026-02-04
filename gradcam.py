"""
Grad-CAM (Gradient-weighted Class Activation Mapping) Implementation
For visualizing which regions of an image contribute most to a model's prediction.
Updated to support Nested Internal Models (e.g. Xception wrapped in Sequential).
"""

import numpy as np
import tensorflow as tf
import cv2

def find_layer_recursive(model, target_layer_name):
    """
    Recursively search for a layer with target_layer_name.
    Returns:
        (inner_model_layer, found_layer, is_nested)
        - inner_model_layer: The layer in the TOP model that contains the target (or None if top-level)
        - found_layer: The actual target layer object
        - is_nested: Boolean
    """
    # 1. Search top level first
    for layer in model.layers:
        if layer.name == target_layer_name:
            return None, layer, False
            
    # 2. Search nested models
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
            # Check inside this layer
            try:
                # We can't easily recurse deeply without keeping track of the path.
                # But for this problem, we know it's likely 1 level deep (User's case).
                # Let's simple check layer.get_layer()
                nested_target = layer.get_layer(target_layer_name)
                return layer, nested_target, True
            except ValueError:
                # Not found in this nested layer, continue
                pass
                
            # If deeper recursion is needed (not implemented to avoid complexity unless needed)
            # For Xception wrapper, it's just 1 level deep: Wrapper -> Xception -> Block14
            pass
            
    return None, None, False

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generate Grad-CAM heatmap. 
    Automatically handles nested models (like Xception wrappers) to avoid Graph Disconnected errors.
    """
    
    # Check if we need nested logic
    inner_model_wrapper, target_layer, is_nested = find_layer_recursive(model, last_conv_layer_name)
    
    if is_nested and inner_model_wrapper is not None:
        print(f"DEBUG: Found nested layer '{last_conv_layer_name}' inside '{inner_model_wrapper.name}'")
        return make_gradcam_heatmap_nested(img_array, model, inner_model_wrapper, last_conv_layer_name, pred_index)
        
    # Standard implementation for top-level layers
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
    except ValueError as e:
        # Fallback if find_layer_recursive missed something but get_layer fails
        raise ValueError(f"Could not find layer '{last_conv_layer_name}'. Error: {e}")

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def make_gradcam_heatmap_nested(img_array, full_model, inner_model_layer, target_layer_name, pred_index=None):
    """
    Robust Grad-CAM for nested models.
    Executes in 3 stages: Pre-process -> Inner Model -> Post-process.
    """
    # 1. Build Pre-processing Model (Full Input -> Inner Model Input)
    # We find the layer *before* the inner model to avoid "Graph Disconnected" on inner_model.input
    
    # Find index of inner model
    layer_index = None
    for i, layer in enumerate(full_model.layers):
        if layer.name == inner_model_layer.name:
            layer_index = i
            break
            
    if layer_index is None:
        raise ValueError(f"Layer {inner_model_layer.name} not found in top-level model.")

    if layer_index == 0:
        # Inner model is the very first layer
        intermediate_input = img_array
    else:
        # Use the output of the previous layer as the bridge
        prev_layer = full_model.layers[layer_index - 1]
        try:
            pre_model = tf.keras.models.Model(
                inputs=full_model.inputs,
                outputs=prev_layer.output
            )
            intermediate_input = pre_model(img_array)
        except Exception as e:
            raise ValueError(f"Could not build pre-model up to {prev_layer.name}: {e}")

    # 2. Build Grad-Sub-Model (Inner Input -> [Target Layer, Inner Output])
    # This enables gradient tracking INSIDE the nested model
    grad_sub_model = tf.keras.models.Model(
        inputs=inner_model_layer.inputs,
        outputs=[inner_model_layer.get_layer(target_layer_name).output, inner_model_layer.output]
    )

    # 3. Identify Post-processing Layers (Layers AFTER the inner model)
    post_layers = []
    found_inner = False
    for layer in full_model.layers:
        if layer.name == inner_model_layer.name:
            found_inner = True
            continue
        if found_inner:
            post_layers.append(layer)

    # 4. Execution Loop
    with tf.GradientTape() as tape:
        # A. Pre-processing is done (intermediate_input)
        
        # B. Run Inner Model (Watch this!)
        # Check if intermediate_input needs watching if it's a tensor? 
        # No, tape filters variables. But since we start tape here, 
        # the operations inside grad_sub_model will be recorded.
        conv_out, inner_features = grad_sub_model(intermediate_input)
        
    # C. Run Post-processing (The Head)
        x = inner_features
        if isinstance(x, list):
            x = x[0]
        
        for layer in post_layers:
            x = layer(x)
        preds = x
        
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # 5. Compute Gradients
    grads = tape.gradient(class_channel, conv_out)
    
    # 6. Generate Heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize the heatmap between 0 & 1 for visualization
    # Add epsilon to avoid NaN if max is 0
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    
    if max_val == 0:
        print("DEBUG: Heatmap max is 0 (ReLU killed everything or empty gradients)")
        return heatmap.numpy() # Returns zeros
        
    heatmap = heatmap / (max_val + 1e-10)
    
    return heatmap.numpy()

def overlay_heatmap(heatmap, original_img, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay the Grad-CAM heatmap on the original image.
    """
    if hasattr(original_img, 'convert'):
        original_img = np.array(original_img)
    
    if original_img.dtype != np.uint8:
        original_img = original_img.astype(np.uint8)
    
    img_height, img_width = original_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    superimposed_img = cv2.addWeighted(
        original_img, 
        1 - alpha, 
        heatmap_colored, 
        alpha, 
        0
    )
    
    return superimposed_img

def find_last_conv_layer(model):
    """
    Finds last conv layer. Unpacks one level of nesting if needed.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
            return layer.name
        # Check inside if it's a model
        if isinstance(layer, tf.keras.Model) or hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D) or 'conv' in sub_layer.name.lower():
                    return sub_layer.name
    raise ValueError("No convolutional layer found!")
