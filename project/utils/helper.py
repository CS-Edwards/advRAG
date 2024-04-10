from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List,Dict, Any


#
# @TODO change
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500, #changed from 2000
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)


def key_to_node_label(key: str) -> str:
    """
    Maps a key to its corresponding node label based on a predefined mapping.

    Args:
        key (str): The key to be mapped to a node label.

    Returns:
        str: The corresponding node label if the key is found in the mapping, otherwise an empty string.
    """
    key_node_map: Dict[str, str] = {
        'Standard_formal': 'STANDARD',
        'Definitions': 'DEFINITIONS',
        'Basis_for_judgement': 'BASIS',
        'Supporting_docs': 'DOCUMENTATION'
    }
    return key_node_map.get(key, '')  # Return empty string if key not found




def split_text_from_node(text: str, key: str) -> List[Dict[str, Any]]:
    """
    Splits the input text into chunks and generates metadata for each chunk.

    Args:
        text (str): The text to be split into chunks.
        key (str): The key representing the type of data.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a chunk
            with metadata including the chunk text, parent type, and chunk sequence ID.
    """
    chunks_with_metadata = []
    chunks = text_splitter.split_text(text)

    for i, chunk in enumerate(chunks):
        #print(f'{i} : {chunk}')
        chunks_with_metadata.append({
            'type': 'dataChunk',
            'text': chunk,
            'parentType': key,
            'chunk_sequence_id': i,
        })
    
    return chunks_with_metadata



def chunk_doc(standard: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes top-level items from the standard dictionary.

    Args:
        standard (Dict[str, Any]): The dictionary containing top-level items.

    Returns:
        Dict[str, Any]: A dictionary where keys are top-level item keys ('Standard_formal',
            'Definitions', 'Basis_for_judgement', 'Supporting_docs'), and values are lists of
            dictionaries representing chunks with metadata for each item.
    """
    chunk_dict = {}

    for key in ['Standard_formal', 'Definitions', 'Basis_for_judgement', 'Supporting_docs']:
        #print(key)
        #print(type(key))
        item_text = standard[key]
        item_label = key_to_node_label(key)
        dict_val = split_text_from_node(item_text, item_label)
        chunk_dict[key] = dict_val

    return chunk_dict
