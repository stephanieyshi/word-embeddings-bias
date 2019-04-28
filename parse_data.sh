while IFS="," read -r a b c d e f g h i j;
do
    tr -d '[:punct:]' <<< "$j" | tr '[:upper:]' '[:lower:]' | awk 'ORS=FS' >> articles_lower.txt
    echo -e >> articles_lower.txt
done < articles1.csv

echo 'article 1 done'

while IFS="," read -r a b c d e f g h i j;
do
    tr -d '[:punct:]' <<< "$j" | tr '[:upper:]' '[:lower:]' | awk 'ORS=FS' >> articles_lower.txt
    echo -e >> articles_lower.txt
done < articles2.csv

echo 'article 2 done'

while IFS="," read -r a b c d e f g h i j;
do
    tr -d '[:punct:]' <<< "$j" | tr '[:upper:]' '[:lower:]' | awk 'ORS=FS' >> articles_lower.txt
    echo -e >> articles_lower.txt
done < articles3.csv

echo 'article 3 done'
