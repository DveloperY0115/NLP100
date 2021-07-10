from torchtext.legacy import data
from torchtext.legacy.data.iterator import batch

# define fields
TITLE = data.Field(
    sequential=True, use_vocab=True, tokenize=str.split, lower=True, batch_first=True, fix_length=20
)

PUBLISHER = data.Field(sequential=False, use_vocab=True, lower=True, batch_first=True,)

CATEGORY = data.Field(sequential=False, use_vocab=True, batch_first=True, is_target=True)

