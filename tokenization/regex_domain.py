import re

pattern = r'^(https?):\/\/([a-zA-Z0-9]{1,26}[-\.]){1,33}\w+\/?$'
result = re.match(pattern, 'http://example.com/')
print(result)
