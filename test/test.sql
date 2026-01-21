SELECT    *
FROM      cranfield;

SELECT    *
FROM      cranfield_df;

SELECT    *
FROM      cranfield_pl;

DROP      TABLE if EXISTS cranfield_pl;

DROP      TABLE if EXISTS cranfield_df;

DROP      TABLE if EXISTS cranfield;

SELECT    *
FROM      cranfield_df
WHERE     term = 'shroud';