def run_pipeline(stages, namespace):
    """Run every stage in order in this one cell; propagate real failures."""
    labels = {
        'catalog':'Load model catalog and process helpers',
        'install':'Install and test the T4 runtime',
        'weights':'Prepare and verify weights',
        'workflow':'Load the render workflow',
        'render':'Sample, decode and verify one clip',
        'result':'Preview and download',
    }
    namespace['RESULT_PATH'] = None
    for index, (name, source) in enumerate(stages.items(), start=1):
        label = labels[name]
        print(f'\n[{index}/{len(stages)}] {label}', flush=True)
        try:
            exec(compile(source, f'<h3:{name}>', 'exec'), namespace)
        except BaseException as error:
            print(f'\nStopped during {label}: {type(error).__name__}: {error}', flush=True)
            print('Later stages were not run. Existing model files and saved latents are retained.', flush=True)
            raise
        print(f'Completed: {label}', flush=True)
    print('\nComplete. The video preview and download are above.', flush=True)
