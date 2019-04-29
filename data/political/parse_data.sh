while IFS="," read -r a b c d e f g h i j;
do
    if [[ $d == 'Breitbart' ]]
    then
      tr -d '[:punct:]' <<< "$j" | tr '[:upper:]' '[:lower:]' | awk 'ORS=FS' >> articles_breitbart_lower.txt
      echo >> articles_breitbart_lower.txt
    fi
done < articles1.csv
