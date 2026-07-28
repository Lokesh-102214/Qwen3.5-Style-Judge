STYLE_MARKERS = {
    "abstract":      ("non-representational geometric forms", "bold flat primary color blocks", "elimination of figurative elements"),
    "acrylic":       ("thick impasto surface texture", "sharp-edged opaque brushwork", "high-saturation vivid colors"),
    "charcoal":      ("monochromatic grey tonal gradients", "smudged dusty texture marks", "visible hatching lines"),
    "cubist":        ("simultaneous multiple-viewpoint planes", "fractured interlocking geometric facets", "muted ochre and brown palette"),
    "expressive":    ("raw emotionally distorted brushwork", "unblended aggressive color strokes", "angular forms"),
    "gouache":       ("opaque chalky flat color areas", "matte non-reflective finish", "clean sharp color-edge definition"),
    "impressionist": ("short broken dappled brushstrokes", "chromatic color juxtaposition without blending", "soft atmospheric hazy edges"),
    "minimalist":    ("extreme negative white-space areas", "single dominant restrained color", "reductive simplified forms"),
    "mosaic":        ("discrete tile fragments with visible grout lines", "angular hard-edge color segmentation", "tesserae patterns"),
    "pencil":        ("parallel graphite hatching lines", "linear tonal gradation from light to dark", "visible paper grain texture"),
    "pop_art":       ("flat bold primary poster colors", "Ben-Day halftone dot patterns", "thick black outline contour lines"),
    "watercolor":    ("transparent layered wash glazes", "soft wet-edge bloom blending", "paper granulation in wash areas"),
}

STYLE_DESCRIPTIONS = {k: ", ".join(v) for k, v in STYLE_MARKERS.items()}

SYSTEM_PROMPT = """You are an elite Visual Stylometric Analyst evaluating a style transfer operation.

Evaluation Task:
Analyze the Query Input → Query Result transformation using the visual DNA established by the provided Exemplar Pairs and Style Markers. Evaluate the success of the transfer using the following three metrics, scoring each on a scale of 1 to 10:

Style Fidelity (1-10): The degree to which the specific physical visual markers from the exemplars are successfully, deeply, and comprehensively applied to the result.
Content Preservation (1-10): The degree to which the original subject, structural layout, and spatial relationships from the Query Input are maintained without hallucination, distortion, or feature bleeding.
Rendering Quality (1-10): The technical execution and coherence of the final image, evaluating the absence of visual artifacts, noise, pixel banding, blur halos, or unnatural seams.

CRITICAL INSTRUCTION FOR STEP 1: You must extract visual markers ONLY from the Perfect Exemplar Pairs. You are strictly forbidden from referencing, describing, or letting the Query Result influence your Step 1 extraction. Define the target baseline strictly before evaluating the transformation.

Always respond in this exact XML structure:
<Thinking_Process>
Step 1: Style Extraction: [Extract 3 physical visual markers STRICTLY and ONLY from the Perfect 10/10 Exemplar Pairs. Do NOT look at or describe the Query Result in this step.]
Step 2: Analogy Check: [Evaluate Query Input -> Query Result strictly against those 3 exact markers extracted in Step 1.]
Step 3: Conflict Check: [Note any structural distortions, style/content conflicts, or rendering artifacts in the Query Result.]
</Thinking_Process>

<Quantitative_Metrics>
Style Fidelity: [integer 1-11]
Content Preservation: [integer 1-11]
Rendering Quality: [integer 1-11]
</Quantitative_Metrics>

<Final_Output>
Brief Rationale: [Analytical justification for the scores. Max 1200 words. Do NOT name the style.]
</Final_Output>"""

IMAGE_SIZE = 1024
MAX_SEQ_LEN = 8196
MAX_NEW_TOKENS = 2048

# We use the HF repo provided by the user
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-VL-0.8B-Instruct"  
DEFAULT_ADAPTER = "GamerQuant/Qwen3.5_0.8b_10kbalanced_k2_lora16"
