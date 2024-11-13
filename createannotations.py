
#Put this part in the main function if annotations need to be created

if result:
        #Lets get chunks for annotations
        chunks = result['sentiment']['all_chunks']
        
        # Manual annotations
        annotation_file, annotations = annotation_handler.create_annotations(
            chunks = chunks,
            ticker = ticker,
            n_samples = 10
        )
        
        if annotations:
            print("\nAnnotation Summary:")
            print(f"Created {len(annotations)} annotations")
            print(f"Saved to: {annotation_file}")
            
            # Now these annotations can be used in cross validation
            # You can load them later using:
            # loaded_annotations = annotation_handler.load_annotations(annotation_file)
        
