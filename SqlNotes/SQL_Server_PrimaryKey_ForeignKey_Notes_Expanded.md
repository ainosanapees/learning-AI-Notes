# SQL Server Interview Handbook – Primary Key, Foreign Key & Identity

## Primary Key vs Foreign Key

| Feature | Primary Key | Foreign Key |
|---|---|---|
| Purpose | Uniquely identifies each row | Creates relationship between tables |
| Duplicate Values | Not allowed | Allowed |
| NULL | Not allowed | Allowed if nullable |
| Per Table | One (can be composite) | Multiple allowed |
| References | Same table | Parent PK or UNIQUE key |
| Integrity | Entity Integrity | Referential Integrity |
| Index | Clustered by default in SQL Server (unless specified) | No automatic index |

---

# Primary Key

## Definition
A Primary Key uniquely identifies every row in a table.

## Rules
- Cannot contain NULL.
- Cannot contain duplicate values.
- One Primary Key per table.
- Can be composite.
- SQL Server allows a table without a PK, but it is not recommended.

## Interview Questions

### Can a table exist without a Primary Key?
Yes.

### Can duplicate records exist without a Primary Key?
Yes.

### Can Primary Key contain NULL?
No.

### Can Primary Key contain duplicate values?
No.

### Can Identity exist without Primary Key?
Yes.

### Can Primary Key exist without Identity?
Yes.

### Is every Identity column a Primary Key?
No.

### Can Identity be manually inserted?
Yes, using:
```sql
SET IDENTITY_INSERT Student ON;
```

---

# Identity

Identity automatically generates values.
Identity != Primary Key.
Identity does not enforce uniqueness.

---

# Foreign Key

## Definition
A Foreign Key creates a relationship between parent and child tables.

## Rules
- Duplicate values are allowed.
- NULL is allowed if column is nullable.
- References Primary Key or UNIQUE key.
- Multiple foreign keys are allowed.

## Parent / Child Example

Parent
```
Student
---------
StudentId (PK)
Name
```

Child
```
Marks
---------
MarkId (PK)
StudentId (FK)
Marks
```

## Why duplicates are allowed?

|MarkId|StudentId(FK)|Subject|
|---:|---:|---|
|1|1|Math|
|2|1|English|
|3|1|Science|

One parent can have many child rows (One-to-Many).

## Can Foreign Key contain NULL?

Yes.

```sql
DepartmentId INT NULL
```

If declared:

```sql
DepartmentId INT NOT NULL
```

NULL is not allowed.

## Can Foreign Key reference only Primary Key?

No.

It can reference:
- Primary Key
- UNIQUE Key

## Is ID column mandatory?

No. Any PK or UNIQUE column can be referenced.

---

# Referential Integrity

Every Foreign Key value must either:
- exist in the parent table, or
- be NULL (if nullable).

Invalid example:

Parent has DepartmentId = 1,2

Trying to insert DepartmentId = 10 into child -> ERROR.

---

# Cascaded Referential Integrity

## ON DELETE CASCADE
Deleting parent automatically deletes child rows.

## ON UPDATE CASCADE
Updating parent key automatically updates child FK values.

Other options:
- ON DELETE SET NULL
- ON DELETE SET DEFAULT
- NO ACTION

---

# Tricky Interview Questions

1. Does Identity automatically make a column a Primary Key? → No.
2. Can a table exist without a Primary Key? → Yes.
3. Can a Foreign Key contain duplicate values? → Yes.
4. Why? → One parent can have many child records.
5. Can a Foreign Key contain NULL? → Yes, if nullable.
6. Can a Foreign Key reference a UNIQUE key? → Yes.
7. Is ID column mandatory? → No.
8. Can child exist without parent? → Only if FK is NULL; otherwise FK value must exist.
9. Can parent be deleted when child exists? → No, unless CASCADE/other referential action.
10. Difference between Identity and Primary Key? → Identity auto-generates values; Primary Key enforces uniqueness.

---

# Common Interview Mistakes

- Identity = Primary Key ❌
- Foreign Key cannot have duplicates ❌
- Every table must have Primary Key ❌
- Foreign Key always references only Primary Key ❌

---

# Quick Revision

- Primary Key = Unique + NOT NULL
- Foreign Key = Relationship
- Identity = Auto Increment
- PK != Identity
- FK duplicates allowed
- FK NULL allowed if nullable
- FK references PK or UNIQUE
- Referential Integrity prevents invalid references
- Cascaded Referential Integrity automatically propagates update/delete operations.
