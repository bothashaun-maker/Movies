"""Native WanGP settings for the two otherwise-identical orbit tests."""
MODEL = 'minimax_h3_ref2va_pruned_pdd'
PROMPT = '''subject_definitions:
<Subject 1> is the complete crucifixion scene in <Picture 1>, including Jesus,
the wooden cross, Roman soldiers, mourners, distant crosses and storm landscape.
summary:
[reference generation] Preserve <Subject 1> while making one gentle front-side camera arc.
retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - match the people, faces, clothing,
cross geometry, body-to-cross attachment, composition, storm clouds and lighting of <Picture 1>.
detailed_description:
[Shot 1] Begin from the same front three-quarter viewpoint and framing as <Picture 1>.
Jesus remains on the camera-facing side of the cross: his face, chest, abdomen,
arms and legs are visible in front of the wood; the upright and crossbar remain behind him.
His hands, feet and body keep the same attachment points and orientation as the reference.
Keep the main crucifix on the left side of the composition at the reference scale.
Preserve the foreground soldiers and mourners, their positions relative to the crucifix,
their costumes and faces, and the two distant crosses on the right hillside.
The camera makes a slow, steady 30-degree left-to-right arc entirely on this same
front side, at constant height and distance with a locked focal length.
Maintain an unobstructed view of Jesus and the crowd, allowing only natural parallax.
Preserve the dark storm lighting and clouds from the reference. Subtle wind moves fabric.
One continuous shot with stable anatomy, stable cross proportions and unchanged identities.
overall_soundscape: Quiet wind and distant rolling thunder.
non_diegetic_music: N/A'''


def shot_settings(guided=False):
    return dict(model_type=MODEL, settings_version=2.76, image_mode=0,
                resolution='864x480', video_length=124, sliding_window_size=124,
                sliding_window_overlap=18, sliding_window_discard_last_frames=0,
                batch_size=1, repeat_generation=1, seed=8675309, force_fps='24',
                num_inference_steps=8, guidance_phases=1, sample_solver='euler',
                guidance_scale=1.0, flow_shift=12.0, image_prompt_type='',
                video_prompt_type='IV-U' if guided else 'I', audio_prompt_type='',
                config='gguf_q4_k_m,bf16', prompt_enhancer='', overrides='',
                remove_background_images_ref=0, image_refs_relative_size=100,
                prompt=PROMPT, negative_prompt='', multi_prompts_gen_type='FG',
                activated_loras=[], loras_multipliers='', skip_steps_cache_type='',
                temporal_upsampling='', spatial_upsampling='', custom_sigmas='',
                output_filename='')


def server_settings(cache, output):
    return dict(last_model_type=MODEL, transformer_types=[MODEL],
                transformer_quantization='int8', text_encoder_quantization='int8',
                transformer_dtype_policy='fp16', attention_mode='sdpa', compile='disabled',
                int8_kernels='disabled', fp8_kernels='pytorch',
                profile=5, video_profile=5, image_profile=5, audio_profile=5,
                perc_reserved_mem_max=10, video_preload_mode='manual',
                video_preload_in_VRAM=0, preload_model_policy=[],
                vram_allocator='default', ram_allocator='default',
                vae_config=3, generation_preview='rgb',
                checkpoints_paths=[str(cache)],
                save_path=str(output/'videos'), image_save_path=str(output/'images'),
                audio_save_path=str(output/'audio'), metadata_type='json',
                video_output_codec='libx264_8', video_output_container='mp4',
                save_queue_if_crash=1, clear_file_list=0,
                process_queues_when_browser_unfocused=0)
