class HumanValidator:
    def __init__(self):
        self.annotation_template = {
            'sentiment': None,  # -1 to 1
            'confidence': None, # 1 to 5
            'key_points': [],   # Important points noted
            'notes': ''         # Any additional notes
        }
    
    def create_validation_task(self, text_chunks):
        """Create validation tasks for human annotators"""
        return [{
            'chunk_id': i,
            'text': chunk,
            'annotation': self.annotation_template.copy()
        } for i, chunk in enumerate(text_chunks)]