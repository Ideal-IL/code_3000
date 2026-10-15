import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_identifiers = [col for col in anon_df.columns
                         if col in aux_df.columns
                         and col not in ['anon_id', 'matched_name', 'name']]
    matches = []
    for _, anon_row in anon_df.iterrows():
        mask = pd.Series(True, index=aux_df.index)

        for col in quasi_identifiers:
            mask &= (aux_df[col] == anon_row[col])

        candidate_matches = aux_df[mask]
        if len(candidate_matches) == 1:
            matches.append({
                'anon_id': anon_row['anon_id'],
                'matched_name': candidate_matches.iloc[0]['name']
            })
    return pd.DataFrame(matches, columns=['anon_id', 'matched_name'])


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    if len(anon_df) == 0:
        return 0.0
    return len(matches_df) / len(anon_df)
